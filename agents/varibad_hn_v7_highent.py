"""V7: High Entropy — 5× stronger entropy bonus to sustain exploration.

Based on V2 (μ-only HN). The default ent_coef=0.01 may be too weak to
prevent the HyperNet-generated policy from collapsing to near-deterministic
early in training. Once entropy dies, the policy can't explore its way
to a better solution during the refinement phase.

Fix: use ent_coef=0.05 (5× default). Forces the policy to maintain
meaningful exploration throughout training, at the cost of slightly
noisier early performance. The hypothesis is that sustained exploration
lets the refinement phase make progress instead of stalling.
"""
from .varibad_hn_v2_mu import V2MuOnlyActorCritic, v2_mu_elbo_loss_fn
from .common import ppo_loss_from_logits
from .rl2_hn import apply_generated_policy

import jax
import jax.numpy as jnp

V7HighEntActorCritic = V2MuOnlyActorCritic
v7_highent_elbo_loss_fn = v2_mu_elbo_loss_fn


def v7_highent_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.05,
                            vf_coef=0.5):
    """Same as V2 PPO loss but with ent_coef=0.05 (5× default)."""
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        new_h = model.gru_cell(x, h)
        post = model.posterior_net(new_h)
        mu = post[:model.latent_dim]
        log_sig = jnp.clip(post[model.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)
        hn_in = jnp.concatenate([new_h, mu])
        policy_params = model.hyper_net(hn_in)
        logits = apply_generated_policy(
            o, policy_params, model.obs_size, model.policy_hidden, model.n_actions)
        critic_in = jnp.concatenate([new_h, mu, sigma, o])
        value = model.critic_head(critic_in).squeeze(-1)
        return logits, value

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
