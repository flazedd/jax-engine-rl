"""V14: Fast Weights — outer-product associative memory replacing GRU.

From Schmidhuber (1992), Ba et al. (2016). The "slow weights" (learned via
gradient descent) learn how to write key-value pairs to a fast-weight
matrix M. M is updated via outer products at each timestep and serves as
differentiable associative memory. The policy queries M with the current
observation to retrieve task-relevant information.

Unlike a GRU hidden state (fixed-size vector that gets overwritten), M
accumulates information in a structured way via key-value associations.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits


class V14FastWeightActorCritic(eqx.Module):
    """Fast-weight actor-critic. Memory via outer-product matrix."""
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    query_proj: eqx.nn.Linear
    actor_head: MLP
    critic_head: MLP
    fw_dim: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)
    decay: float = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 fw_dim=8, decay=0.95, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.fw_dim = fw_dim
        self.hidden_size = fw_dim * fw_dim
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.decay = decay

        self.key_proj = eqx.nn.Linear(input_size, fw_dim, key=k1)
        self.value_proj = eqx.nn.Linear(input_size, fw_dim, key=k2)
        self.query_proj = eqx.nn.Linear(obs_size, fw_dim, key=k3)

        head_input = fw_dim + obs_size
        self.actor_head = MLP([head_input, 64, n_actions], key=k4)
        self.critic_head = MLP([head_input, 64, 1], key=k5)

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def forward_step(self, aug_input, obs, hidden):
        M = hidden.reshape(self.fw_dim, self.fw_dim)

        # Write: key-value pair from augmented input
        k = jax.nn.tanh(self.key_proj(aug_input))
        v = jax.nn.tanh(self.value_proj(aug_input))
        M_new = self.decay * M + jnp.outer(k, v)

        # Read: query from current observation
        q = self.query_proj(obs)
        context = M_new @ q
        context = context / (jnp.linalg.norm(context) + 1e-6)

        head_in = jnp.concatenate([context, obs])
        logits = self.actor_head(head_in)
        val = self.critic_head(head_in).squeeze(-1)
        return logits, val, M_new.ravel()


def v14_fw_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        M = h.reshape(model.fw_dim, model.fw_dim)
        k = jax.nn.tanh(model.key_proj(x))
        v = jax.nn.tanh(model.value_proj(x))
        M_new = model.decay * M + jnp.outer(k, v)
        q = model.query_proj(o)
        context = M_new @ q
        context = context / (jnp.linalg.norm(context) + 1e-6)
        head_in = jnp.concatenate([context, o])
        logits = model.actor_head(head_in)
        val = model.critic_head(head_in).squeeze(-1)
        return logits, val

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
