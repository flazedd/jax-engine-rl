"""V16: Attention — cross-attention over episode transition buffer.

Instead of compressing history into a fixed-size GRU hidden state, store
past (obs, prev_act, prev_rew) tuples in a fixed-size circular buffer.
At each step, cross-attention retrieves relevant past transitions:
query = current observation, keys/values = buffer entries.

For task identification from noisy side signals, attention can selectively
retrieve the most informative past observations rather than relying on
the GRU's compressed representation.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits


class V16AttentionActorCritic(eqx.Module):
    """Attention-buffer actor-critic. Memory via cross-attention."""
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    query_proj: eqx.nn.Linear
    actor_head: MLP
    critic_head: MLP
    buffer_size: int = eqx.field(static=True)
    entry_size: int = eqx.field(static=True)
    attn_dim: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 buffer_size=10, attn_dim=16, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.buffer_size = buffer_size
        self.entry_size = input_size
        self.attn_dim = attn_dim
        self.hidden_size = buffer_size * input_size + 1
        self.obs_size = obs_size
        self.n_actions = n_actions

        self.key_proj = eqx.nn.Linear(input_size, attn_dim, key=k1)
        self.value_proj = eqx.nn.Linear(input_size, attn_dim, key=k2)
        self.query_proj = eqx.nn.Linear(obs_size, attn_dim, key=k3)

        head_input = attn_dim + obs_size
        self.actor_head = MLP([head_input, 64, n_actions], key=k4)
        self.critic_head = MLP([head_input, 64, 1], key=k5)

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def forward_step(self, aug_input, obs, hidden):
        buf = hidden[:-1].reshape(self.buffer_size, self.entry_size)
        write_idx = jnp.int32(jnp.round(hidden[-1]))

        # Write current input (circular buffer)
        pos = write_idx % self.buffer_size
        buf = buf.at[pos].set(aug_input)
        new_idx = write_idx + 1
        n_valid = jnp.minimum(new_idx, self.buffer_size)

        # Cross-attention
        query = self.query_proj(obs)                          # (attn_dim,)
        keys = jax.vmap(self.key_proj)(buf)                   # (B, attn_dim)
        values = jax.vmap(self.value_proj)(buf)               # (B, attn_dim)

        scores = keys @ query / jnp.sqrt(jnp.float32(self.attn_dim))
        mask = jnp.arange(self.buffer_size) < n_valid
        scores = jnp.where(mask, scores, -1e9)
        weights = jax.nn.softmax(scores)
        context = weights @ values                             # (attn_dim,)

        head_in = jnp.concatenate([context, obs])
        logits = self.actor_head(head_in)
        val = self.critic_head(head_in).squeeze(-1)

        new_hidden = jnp.concatenate([buf.ravel(),
                                       jnp.float32(new_idx)[None]])
        return logits, val, new_hidden


def v16_attention_loss_fn(model, batch,
                          clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        buf = h[:-1].reshape(model.buffer_size, model.entry_size)
        write_idx = jnp.int32(jnp.round(h[-1]))
        pos = write_idx % model.buffer_size
        buf = buf.at[pos].set(x)
        new_idx = write_idx + 1
        n_valid = jnp.minimum(new_idx, model.buffer_size)

        query = model.query_proj(o)
        keys = jax.vmap(model.key_proj)(buf)
        vals = jax.vmap(model.value_proj)(buf)
        scores = keys @ query / jnp.sqrt(jnp.float32(model.attn_dim))
        mask = jnp.arange(model.buffer_size) < n_valid
        scores = jnp.where(mask, scores, -1e9)
        weights = jax.nn.softmax(scores)
        context = weights @ vals

        head_in = jnp.concatenate([context, o])
        logits = model.actor_head(head_in)
        val = model.critic_head(head_in).squeeze(-1)
        return logits, val

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
