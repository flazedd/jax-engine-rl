"""V22: Posterior Velocity — HN conditioned on (h, μ, Δμ).

Feeds both the current belief μ_t AND its temporal derivative
Δμ_t = μ_t - μ_{t-1} to the HyperNetwork. The velocity signal tells
the policy whether the belief is stabilizing (exploit) or shifting
(explore more). Particularly useful for active sensing where the
explore-exploit tradeoff depends on confidence dynamics.

Hidden state = [gru_h | prev_mu]
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params


class V22PosteriorVelActorCritic(eqx.Module):
    gru_cell: eqx.nn.GRUCell
    posterior_net: MLP
    decoder_net: MLP
    hyper_net: MLP
    critic_head: MLP
    gru_hidden_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 gru_hidden=64, latent_dim=4, policy_hidden=16, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.gru_hidden_size = gru_hidden
        self.hidden_size = gru_hidden + latent_dim  # gru_h + prev_mu
        self.latent_dim = latent_dim
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions

        self.gru_cell = eqx.nn.GRUCell(input_size, gru_hidden, key=k1)

        post = MLP([gru_hidden, 64, 2 * latent_dim], key=k2)
        self.posterior_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), post,
            (jnp.zeros_like(post.layers[-1].weight),
             jnp.zeros_like(post.layers[-1].bias)))

        decoder_input = latent_dim + obs_size + n_actions
        self.decoder_net = MLP([decoder_input, 64, 1 + obs_size], key=k3)

        # HN input: h + μ + Δμ
        hn_input = gru_hidden + 2 * latent_dim
        n_params = compute_n_policy_params(obs_size, policy_hidden, n_actions)
        hn = MLP([hn_input, 256, n_params], key=k4)
        self.hyper_net = eqx.tree_at(
            lambda m: m.layers[-1].weight, hn,
            jnp.zeros_like(hn.layers[-1].weight))

        critic_input = gru_hidden + 2 * latent_dim + obs_size
        self.critic_head = MLP([critic_input, 64, 1], key=k5)

    def _forward(self, aug_input, obs, hidden):
        gru_h = hidden[:self.gru_hidden_size]
        prev_mu = hidden[self.gru_hidden_size:]

        new_h = self.gru_cell(aug_input, gru_h)
        post = self.posterior_net(new_h)
        mu = post[:self.latent_dim]
        log_sig = jnp.clip(post[self.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)
        delta_mu = mu - prev_mu

        hn_in = jnp.concatenate([new_h, mu, delta_mu])
        policy_params = self.hyper_net(hn_in)
        logits = apply_generated_policy(
            obs, policy_params, self.obs_size, self.policy_hidden, self.n_actions)

        critic_in = jnp.concatenate([new_h, mu, sigma, obs])
        val = self.critic_head(critic_in).squeeze(-1)

        new_hidden = jnp.concatenate([new_h, mu])
        return logits, val, new_hidden

    def forward_step(self, aug_input, obs, hidden):
        return self._forward(aug_input, obs, hidden)


def v22_vel_elbo_loss_fn(model, elbo_data, rng_key, beta=1.0):
    """Standard ELBO using gru_hidden_size for GRU init."""
    obs, actions, rewards, next_obs, prev_act_oh, prev_rew = elbo_data
    n_steps, n_trials = obs.shape[:2]
    eps = jax.random.normal(rng_key, (n_steps, n_trials, model.latent_dim))

    def forward_trial(obs_s, act_s, rew_s, nobs_s, pa_s, pr_s, eps_s):
        def scan_fn(h, inp):
            o, pa, pr, e = inp
            aug = jnp.concatenate([o, pa, pr])
            h = model.gru_cell(aug, h)
            post = model.posterior_net(h)
            mu = post[:model.latent_dim]
            log_sig = jnp.clip(post[model.latent_dim:], -5.0, 2.0)
            z = mu + jnp.exp(log_sig) * e
            return h, (mu, log_sig, z)

        h0 = jnp.zeros(model.gru_hidden_size)
        _, (mus, log_sigs, zs) = jax.lax.scan(
            scan_fn, h0, (obs_s, pa_s, pr_s, eps_s))

        act_oh = jax.nn.one_hot(act_s, model.n_actions)
        dec_in = jnp.concatenate([zs, obs_s, act_oh], axis=-1)
        dec_out = jax.vmap(model.decoder_net)(dec_in)
        r_hat = dec_out[:, 0]
        o_hat = dec_out[:, 1:]
        recon_r = jnp.mean((r_hat - rew_s) ** 2)
        recon_o = jnp.mean((o_hat - nobs_s) ** 2)
        kl = 0.5 * jnp.mean(jnp.sum(
            mus ** 2 + jnp.exp(2 * log_sigs) - 1 - 2 * log_sigs, axis=-1))
        return recon_r + recon_o + kl

    trial_losses = jax.vmap(forward_trial, in_axes=(1, 1, 1, 1, 1, 1, 1))(
        obs, actions, rewards, next_obs, prev_act_oh, prev_rew, eps)
    return beta * jnp.mean(trial_losses)


def v22_vel_ppo_loss_fn(model, batch,
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
