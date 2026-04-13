"""VariBAD — VAE-based meta-RL with explicit Bayesian inference."""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP


class VariBADActorCritic(eqx.Module):
    """VariBAD architecture.

    Encoder: GRU over (obs, prev_action_onehot, prev_reward) → h_t
    Posterior: MLP(h_t) → (μ_t, log σ_t), latent_dim dimensions
    Decoder: MLP(z_t, obs_t, action_onehot_t) → (reward_hat, next_obs_hat)
    Policy: MLP(obs_t, μ_t, σ_t) → action logits
    Critic: MLP(obs_t, μ_t, σ_t) → value
    """
    gru_cell: eqx.nn.GRUCell
    posterior_net: MLP
    decoder_net: MLP
    actor_head: MLP
    critic_head: MLP
    hidden_size: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 hidden_size=128, latent_dim=4, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.obs_size = obs_size
        self.n_actions = n_actions

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=k1)

        # Posterior: h_t → (μ, log σ). Last layer zeroed → N(0,1) prior at init
        post = MLP([hidden_size, 64, 2 * latent_dim], key=k2)
        self.posterior_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), post,
            (jnp.zeros_like(post.layers[-1].weight),
             jnp.zeros_like(post.layers[-1].bias)))

        # Decoder: (z, obs, action_oh) → (reward_hat, next_obs_hat)
        decoder_input = latent_dim + obs_size + n_actions
        decoder_output = 1 + obs_size
        self.decoder_net = MLP([decoder_input, 64, decoder_output], key=k3)

        # Policy & critic: (obs, μ, σ) → logits / value
        policy_input = obs_size + 2 * latent_dim
        self.actor_head = MLP([policy_input, 64, n_actions], key=k4)
        self.critic_head = MLP([policy_input, 64, 1], key=k5)

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def forward_step(self, aug_input, obs, hidden):
        """Single step → (logits, value, new_hidden, mu, sig)."""
        new_h = self.gru_cell(aug_input, hidden)
        post = self.posterior_net(new_h)
        mu = post[:self.latent_dim]
        log_sig = jnp.clip(post[self.latent_dim:], -5.0, 2.0)
        sig = jnp.exp(log_sig)
        pi_in = jnp.concatenate([obs, mu, sig])
        logits = self.actor_head(pi_in)
        value = self.critic_head(pi_in).squeeze(-1)
        return logits, value, new_h


# ---------------------------------------------------------------------------
# ELBO loss (full trajectory — trains GRU + posterior + decoder)
# ---------------------------------------------------------------------------

def elbo_loss_fn(model, elbo_data, rng_key, beta=1.0):
    """ELBO loss over full trial trajectories.

    elbo_data: (obs, actions, rewards, next_obs, prev_act_oh, prev_rew)
               each shaped (n_steps, n_trials, ...)
    """
    obs, actions, rewards, next_obs, prev_act_oh, prev_rew = elbo_data

    n_steps = obs.shape[0]
    n_trials = obs.shape[1]
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

        h0 = jnp.zeros(model.hidden_size)
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


# ---------------------------------------------------------------------------
# PPO loss (mini-batched with stored GRU hidden states)
# ---------------------------------------------------------------------------

def varibad_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    """Combined PPO loss for VariBAD.

    batch: (obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns)
    """
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch

    def forward_one(o, pa, pr, h):
        aug = jnp.concatenate([o, pa, pr])
        new_h = model.gru_cell(aug, h)
        post = model.posterior_net(new_h)
        mu = post[:model.latent_dim]
        log_sig = jnp.clip(post[model.latent_dim:], -5.0, 2.0)
        sig = jnp.exp(log_sig)
        pi_in = jnp.concatenate([o, mu, sig])
        logits = model.actor_head(pi_in)
        value = model.critic_head(pi_in).squeeze(-1)
        return logits, value

    logits, values = jax.vmap(forward_one)(obs, prev_act, prev_rew, gru_h)

    lp_all = jax.nn.log_softmax(logits)
    lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)
    ratio = jnp.exp(lp - old_lp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    actor_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * lp_all, axis=-1))
    critic_loss = jnp.mean((values - returns) ** 2)

    return actor_loss - ent_coef * entropy + vf_coef * critic_loss
