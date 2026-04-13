"""PPO MLP — actor-critic with separate networks."""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP


class ActorCritic(eqx.Module):
    """MLP actor-critic.  Input: raw observation."""
    actor: MLP
    critic: MLP

    def __init__(self, obs_size, n_actions, hidden=64, *, key):
        k1, k2 = jax.random.split(key)
        self.actor = MLP([obs_size, hidden, hidden, n_actions], key=k1)
        self.critic = MLP([obs_size, hidden, hidden, 1], key=k2)

    def __call__(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def actor_loss_fn(actor, critic, batch, clip_eps=0.2, ent_coef=0.01):
    """PPO clipped actor loss with entropy bonus.

    batch: (obs, actions, old_lp, advantages, returns)
    """
    obs, actions, old_lp, advantages, _ = batch
    logits = jax.vmap(actor)(obs)
    lp_all = jax.nn.log_softmax(logits)
    lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)
    ratio = jnp.exp(lp - old_lp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * lp_all, axis=-1))
    return loss - ent_coef * entropy


def critic_loss_fn(critic, batch):
    """MSE value loss.

    batch: (obs, actions, old_lp, advantages, returns)
    """
    obs, _, _, _, returns = batch
    vals = jax.vmap(critic)(obs).squeeze(-1)
    return jnp.mean((vals - returns) ** 2)
