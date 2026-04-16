"""V8: Conservative PPO — smaller clip range + higher entropy for HN stability.

Based on V2 (μ-only HN). HyperNets amplify parameter updates: a small
change in the latent z produces correlated changes across all generated
policy weights simultaneously. PPO's clip ratio (default ε=0.2) assumes
moderate per-step policy changes, but HN updates can blow past this,
causing most samples to clip and effective gradients to vanish.

Fix: clip_eps=0.1 (half default) + ent_coef=0.03 (3× default).
The tighter clip prevents overshooting, while the higher entropy
maintains exploration. Together they trade slightly slower early
progress for more stable late-stage refinement.
"""
from .varibad_hn_v2_mu import V2MuOnlyActorCritic, v2_mu_elbo_loss_fn
from .common import ppo_loss_from_logits
from .rl2_hn import apply_generated_policy

import jax
import jax.numpy as jnp

V8ConservativeActorCritic = V2MuOnlyActorCritic
v8_conservative_elbo_loss_fn = v2_mu_elbo_loss_fn


def v8_conservative_ppo_loss_fn(model, batch, clip_eps=0.1, ent_coef=0.03,
                                 vf_coef=0.5):
    """V2 PPO loss with clip_eps=0.1, ent_coef=0.03."""
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
