"""RL² — GRU actor-critic with observation skip connection."""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP


class GRUActorCritic(eqx.Module):
    """GRU-based actor-critic for RL².

    Input at each step: (obs, prev_action_onehot, prev_reward) concatenated.
    GRU hidden state persists across episode boundaries within a trial.
    Actor and critic heads receive [gru_hidden, obs] (skip connection).
    """
    gru_cell: eqx.nn.GRUCell
    actor_head: MLP
    critic_head: MLP
    hidden_size: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions, hidden_size=128, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.hidden_size = hidden_size
        self.obs_size = obs_size
        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=k1)
        head_input = hidden_size + obs_size
        self.actor_head = MLP([head_input, 64, n_actions], key=k2)
        self.critic_head = MLP([head_input, 64, 1], key=k3)

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def forward_step(self, aug_input, obs, hidden):
        """Single GRU step → (logits, value, new_hidden)."""
        new_h = self.gru_cell(aug_input, hidden)
        head_in = jnp.concatenate([new_h, obs])
        logits = self.actor_head(head_in)
        value = self.critic_head(head_in).squeeze(-1)
        return logits, value, new_h


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def rl2_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    """Combined PPO loss — per-timestep with stored GRU hidden states.

    batch: (obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns)
    """
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch

    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)
    new_h = jax.vmap(model.gru_cell)(aug, gru_h)
    head_in = jnp.concatenate([new_h, obs], axis=-1)
    logits = jax.vmap(model.actor_head)(head_in)
    values = jax.vmap(model.critic_head)(head_in).squeeze(-1)

    # Actor loss
    lp_all = jax.nn.log_softmax(logits)
    lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)
    ratio = jnp.exp(lp - old_lp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    actor_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * lp_all, axis=-1))

    # Critic loss
    critic_loss = jnp.mean((values - returns) ** 2)

    return actor_loss - ent_coef * entropy + vf_coef * critic_loss
