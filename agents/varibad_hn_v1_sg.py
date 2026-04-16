"""V1: Stop-Gradient — ELBO shapes posterior, PPO reads it without distorting.

Reuses VariBADHNActorCritic model. Different PPO loss that stop-gradients
μ/σ before they enter the HyperNet and critic, so PPO cannot push gradients
back through the posterior net. The ELBO is the sole signal shaping the
posterior; the HyperNet learns to consume it.
"""
import jax
import jax.numpy as jnp

from .rl2_hn import apply_generated_policy
from .common import ppo_loss_from_logits
from .varibad_hn import VariBADHNActorCritic, varibad_hn_elbo_loss_fn

# Re-export model and ELBO unchanged
V1SGActorCritic = VariBADHNActorCritic
v1_sg_elbo_loss_fn = varibad_hn_elbo_loss_fn


def v1_sg_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    """PPO loss with stop-gradient on posterior → HN/critic."""
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        new_h = model.gru_cell(x, h)
        post = model.posterior_net(new_h)
        mu = post[:model.latent_dim]
        log_sig = jnp.clip(post[model.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)
        # Stop gradient: ELBO alone shapes the posterior
        mu_sg = jax.lax.stop_gradient(mu)
        sigma_sg = jax.lax.stop_gradient(sigma)
        hn_in = jnp.concatenate([new_h, mu_sg, sigma_sg])
        policy_params = model.hyper_net(hn_in)
        logits = apply_generated_policy(
            o, policy_params, model.obs_size, model.policy_hidden, model.n_actions)
        critic_in = jnp.concatenate([new_h, mu_sg, sigma_sg, o])
        value = model.critic_head(critic_in).squeeze(-1)
        return logits, value

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
