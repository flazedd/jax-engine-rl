"""AMAGO-HN — minimal causal-transformer encoder + HN policy.

Lightweight adaptation of AMAGO (Grigsby et al. 2023) into this codebase's
on-policy PPO framework. The distinguishing feature is the sequence model:
a causal Transformer over a sliding window of the last K tokens replaces
the GRU used by RL²/RL²+HN. Everything else (HN policy head, critic skip,
PPO loss) mirrors RL²+HN so the ablation is clean.

Hidden state carried through the rollout scan is the *token buffer*
shape (K, d_model): shifted each step, newest token at position K-1.
Transformer runs causally over the full buffer each step; output at the
last position conditions the HN and the critic.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP
from .rl2_hn import apply_generated_policy, compute_n_policy_params


def _causal_mask(K):
    # True means "keep", False means "mask" (−inf in logits).
    i = jnp.arange(K)[:, None]
    j = jnp.arange(K)[None, :]
    return j <= i


class CausalSelfAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    n_heads: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)

    def __init__(self, d_model, n_heads, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.q_proj = eqx.nn.Linear(d_model, d_model, key=k1)
        self.k_proj = eqx.nn.Linear(d_model, d_model, key=k2)
        self.v_proj = eqx.nn.Linear(d_model, d_model, key=k3)
        self.out_proj = eqx.nn.Linear(d_model, d_model, key=k4)

    def __call__(self, x):
        # x: (K, d_model)
        K = x.shape[0]
        q = jax.vmap(self.q_proj)(x).reshape(K, self.n_heads, self.d_head)
        k = jax.vmap(self.k_proj)(x).reshape(K, self.n_heads, self.d_head)
        v = jax.vmap(self.v_proj)(x).reshape(K, self.n_heads, self.d_head)
        # (H, K, D_head)
        q = jnp.transpose(q, (1, 0, 2))
        k = jnp.transpose(k, (1, 0, 2))
        v = jnp.transpose(v, (1, 0, 2))
        # (H, K, K)
        scores = jnp.einsum("hid,hjd->hij", q, k) / jnp.sqrt(self.d_head)
        mask = _causal_mask(K)
        scores = jnp.where(mask[None, :, :], scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        # (H, K, D_head) → (K, H, D_head) → (K, d_model)
        out = jnp.einsum("hij,hjd->hid", attn, v)
        out = jnp.transpose(out, (1, 0, 2)).reshape(K, self.d_model)
        return jax.vmap(self.out_proj)(out)


class TransformerBlock(eqx.Module):
    attn: CausalSelfAttention
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear

    def __init__(self, d_model, n_heads, d_ff, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.attn = CausalSelfAttention(d_model, n_heads, key=k1)
        self.ln1 = eqx.nn.LayerNorm(d_model)
        self.ln2 = eqx.nn.LayerNorm(d_model)
        self.ff1 = eqx.nn.Linear(d_model, d_ff, key=k2)
        self.ff2 = eqx.nn.Linear(d_ff, d_model, key=k3)

    def __call__(self, x):
        # Pre-LN transformer block.
        x = x + self.attn(jax.vmap(self.ln1)(x))
        h = jax.vmap(self.ln2)(x)
        h = jax.vmap(self.ff1)(h)
        h = jax.nn.gelu(h)
        h = jax.vmap(self.ff2)(h)
        return x + h


class AMAGOHNActorCritic(eqx.Module):
    token_proj: eqx.nn.Linear
    pos_embed: jnp.ndarray
    block: TransformerBlock
    final_ln: eqx.nn.LayerNorm
    hyper_net: MLP
    critic_head: MLP
    context_len: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 context_len=64, d_model=64, n_heads=4,
                 d_ff=128, policy_hidden=16, *, key):
        k_tok, k_pos, k_blk, k_hn, k_cr = jax.random.split(key, 5)
        self.context_len = context_len
        self.d_model = d_model
        self.hidden_size = d_model
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions

        self.token_proj = eqx.nn.Linear(input_size, d_model, key=k_tok)
        self.pos_embed = 0.02 * jax.random.normal(k_pos, (context_len, d_model))
        self.block = TransformerBlock(d_model, n_heads, d_ff, key=k_blk)
        self.final_ln = eqx.nn.LayerNorm(d_model)

        n_params = compute_n_policy_params(obs_size, policy_hidden, n_actions)
        hn = MLP([d_model, 256, n_params], key=k_hn)
        # Bias-HyperInit: zero last-layer weights → identical base policy at init.
        self.hyper_net = eqx.tree_at(
            lambda m: m.layers[-1].weight, hn,
            jnp.zeros_like(hn.layers[-1].weight))

        self.critic_head = MLP([d_model + obs_size, 64, 1], key=k_cr)

    def init_state(self):
        # Sliding window of projected tokens; zeros for "before trial started".
        return jnp.zeros((self.context_len, self.d_model))

    def _encode(self, buffer_in, aug_input):
        # Project new token, shift buffer left, append at last position.
        new_token = self.token_proj(aug_input)
        new_buffer = jnp.concatenate(
            [buffer_in[1:], new_token[None, :]], axis=0)
        x = new_buffer + self.pos_embed
        x = self.block(x)
        x = jax.vmap(self.final_ln)(x)
        last_h = x[-1]
        return last_h, new_buffer

    def forward_step(self, aug_input, obs, buffer_in):
        """Single step → (logits, value, new_buffer)."""
        last_h, new_buffer = self._encode(buffer_in, aug_input)
        policy_params = self.hyper_net(last_h)
        logits = apply_generated_policy(
            obs, policy_params, self.obs_size,
            self.policy_hidden, self.n_actions)
        critic_in = jnp.concatenate([last_h, obs])
        value = self.critic_head(critic_in).squeeze(-1)
        return logits, value, new_buffer


def amago_hn_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.05, vf_coef=0.5):
    """PPO loss for AMAGO-HN. Higher default entropy (sparse-reward envs)."""
    obs, prev_act, prev_rew, buffer_in, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, b):
        last_h, _ = model._encode(b, x)
        policy_params = model.hyper_net(last_h)
        logits = apply_generated_policy(
            o, policy_params, model.obs_size,
            model.policy_hidden, model.n_actions)
        critic_in = jnp.concatenate([last_h, o])
        value = model.critic_head(critic_in).squeeze(-1)
        return logits, value

    logits, values = jax.vmap(forward_one)(aug, obs, buffer_in)

    lp_all = jax.nn.log_softmax(logits)
    lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)
    ratio = jnp.exp(lp - old_lp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    actor_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * lp_all, axis=-1))
    critic_loss = jnp.mean((values - returns) ** 2)

    return actor_loss - ent_coef * entropy + vf_coef * critic_loss
