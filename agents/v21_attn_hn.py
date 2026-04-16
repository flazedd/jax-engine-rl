"""V21: Attention-conditioned HN — V16 attention + RL²+HN fusion.

Attention over past transitions retrieves the most informative observations.
GRU provides sequential compression. HN generates policy from both signals.
Combines selective memory retrieval with expressive policy conditioning.

Hidden state = [gru_h | buffer_flat | write_idx]
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params


class V21AttnHNActorCritic(eqx.Module):
    gru_cell: eqx.nn.GRUCell
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    query_proj: eqx.nn.Linear
    hyper_net: MLP
    critic_head: MLP
    gru_hidden_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    buffer_size: int = eqx.field(static=True)
    entry_size: int = eqx.field(static=True)
    attn_dim: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 gru_hidden=64, buffer_size=10, attn_dim=16,
                 policy_hidden=16, *, key):
        keys = jax.random.split(key, 7)
        self.gru_hidden_size = gru_hidden
        self.buffer_size = buffer_size
        self.entry_size = input_size
        self.attn_dim = attn_dim
        self.hidden_size = gru_hidden + buffer_size * input_size + 1
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions

        self.gru_cell = eqx.nn.GRUCell(input_size, gru_hidden, key=keys[0])
        self.key_proj = eqx.nn.Linear(input_size, attn_dim, key=keys[1])
        self.value_proj = eqx.nn.Linear(input_size, attn_dim, key=keys[2])
        self.query_proj = eqx.nn.Linear(obs_size, attn_dim, key=keys[3])

        # HN conditioned on (gru_h, attention_context)
        hn_input = gru_hidden + attn_dim
        n_params = compute_n_policy_params(obs_size, policy_hidden, n_actions)
        hn = MLP([hn_input, 256, n_params], key=keys[4])
        self.hyper_net = eqx.tree_at(
            lambda m: m.layers[-1].weight, hn,
            jnp.zeros_like(hn.layers[-1].weight))

        self.critic_head = MLP([gru_hidden + attn_dim + obs_size, 64, 1],
                               key=keys[5])

    def _forward(self, aug_input, obs, hidden):
        gru_h = hidden[:self.gru_hidden_size]
        buf_idx = hidden[self.gru_hidden_size:]
        buf = buf_idx[:-1].reshape(self.buffer_size, self.entry_size)
        write_idx = jnp.int32(jnp.round(buf_idx[-1]))

        new_h = self.gru_cell(aug_input, gru_h)

        # Write + attention
        pos = write_idx % self.buffer_size
        buf = buf.at[pos].set(aug_input)
        new_idx = write_idx + 1
        n_valid = jnp.minimum(new_idx, self.buffer_size)

        query = self.query_proj(obs)
        keys = jax.vmap(self.key_proj)(buf)
        vals = jax.vmap(self.value_proj)(buf)
        scores = keys @ query / jnp.sqrt(jnp.float32(self.attn_dim))
        mask = jnp.arange(self.buffer_size) < n_valid
        scores = jnp.where(mask, scores, -1e9)
        context = jax.nn.softmax(scores) @ vals

        # HN from (gru_h, attention_context)
        hn_in = jnp.concatenate([new_h, context])
        policy_params = self.hyper_net(hn_in)
        logits = apply_generated_policy(
            obs, policy_params, self.obs_size, self.policy_hidden, self.n_actions)

        critic_in = jnp.concatenate([new_h, context, obs])
        val = self.critic_head(critic_in).squeeze(-1)

        new_hidden = jnp.concatenate([new_h, buf.ravel(),
                                       jnp.float32(new_idx)[None]])
        return logits, val, new_hidden

    def forward_step(self, aug_input, obs, hidden):
        return self._forward(aug_input, obs, hidden)


def v21_attn_hn_loss_fn(model, batch,
                        clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        logits, val, _ = model._forward(x, o, h)
        return logits, val

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
