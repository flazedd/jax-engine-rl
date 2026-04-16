"""VariBAD+HN — VAE-based meta-RL with HyperNetwork-generated policy."""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP
from .rl2_hn import apply_generated_policy, compute_n_policy_params


class VariBADHNActorCritic(eqx.Module):
    """VariBAD + HyperNetwork architecture.

    Encoder: GRU over (obs, prev_action_onehot, prev_reward) -> h_t
    Posterior: MLP(h_t) -> (mu, log_sigma), for ELBO training
    Decoder: MLP(z, obs, action_oh) -> (reward_hat, next_obs_hat)
    HyperNet: (h_t, mu_t, sigma_t) -> policy MLP weights (posterior-conditioned)
    Generated policy: obs -> logits
    Critic: [h_t, mu_t, sigma_t, obs] -> value

    Key design: the posterior (μ, σ) flows into both HyperNet and critic,
    so ELBO gradients that shape the posterior directly affect policy
    generation — unlike the original design where h_t alone fed the HyperNet.
    """
    gru_cell: eqx.nn.GRUCell
    posterior_net: MLP
    decoder_net: MLP
    hyper_net: MLP
    critic_head: MLP
    hidden_size: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 hidden_size=128, latent_dim=4, policy_hidden=64, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=k1)

        # Posterior: h_t -> (mu, log_sig). Zeroed last layer -> N(0,1) prior
        post = MLP([hidden_size, 64, 2 * latent_dim], key=k2)
        self.posterior_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), post,
            (jnp.zeros_like(post.layers[-1].weight),
             jnp.zeros_like(post.layers[-1].bias)))

        # Decoder: (z, obs, action_oh) -> (reward_hat, next_obs_hat)
        decoder_input = latent_dim + obs_size + n_actions
        decoder_output = 1 + obs_size
        self.decoder_net = MLP([decoder_input, 64, decoder_output], key=k3)

        # HyperNet: (h_t, mu_t, sigma_t) -> flat policy params
        # Input: h_t (hidden_size) + mu (latent_dim) + sigma (latent_dim)
        hn_input_size = hidden_size + 2 * latent_dim
        n_params = compute_n_policy_params(obs_size, policy_hidden, n_actions)
        hn = MLP([hn_input_size, 256, n_params], key=k4)
        self.hyper_net = eqx.tree_at(
            lambda m: m.layers[-1].weight, hn,
            jnp.zeros_like(hn.layers[-1].weight))

        # Critic: [h_t, mu_t, sigma_t, obs] -> value
        critic_input_size = hidden_size + 2 * latent_dim + obs_size
        self.critic_head = MLP([critic_input_size, 64, 1], key=k5)

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def _get_posterior(self, h):
        """Compute posterior (mu, sigma) from GRU hidden state."""
        post = self.posterior_net(h)
        mu = post[:self.latent_dim]
        log_sig = jnp.clip(post[self.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)
        return mu, sigma

    def forward_step(self, aug_input, obs, hidden):
        """Single step -> (logits, value, new_hidden)."""
        new_h = self.gru_cell(aug_input, hidden)
        mu, sigma = self._get_posterior(new_h)
        hn_in = jnp.concatenate([new_h, mu, sigma])
        policy_params = self.hyper_net(hn_in)
        logits = apply_generated_policy(
            obs, policy_params, self.obs_size, self.policy_hidden, self.n_actions)
        critic_in = jnp.concatenate([new_h, mu, sigma, obs])
        value = self.critic_head(critic_in).squeeze(-1)
        return logits, value, new_h


# ---------------------------------------------------------------------------
# ELBO loss (trains GRU + posterior + decoder)
# ---------------------------------------------------------------------------

def varibad_hn_elbo_loss_fn(model, elbo_data, rng_key, beta=1.0):
    """ELBO loss over full trial trajectories."""
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
# PPO loss (uses HyperNet for policy, not posterior)
# ---------------------------------------------------------------------------

def varibad_hn_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    """PPO loss for VariBAD+HN. Policy via posterior-conditioned HyperNet."""
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        new_h = model.gru_cell(x, h)
        post = model.posterior_net(new_h)
        mu = post[:model.latent_dim]
        log_sig = jnp.clip(post[model.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)
        hn_in = jnp.concatenate([new_h, mu, sigma])
        policy_params = model.hyper_net(hn_in)
        logits = apply_generated_policy(
            o, policy_params, model.obs_size, model.policy_hidden, model.n_actions)
        critic_in = jnp.concatenate([new_h, mu, sigma, o])
        value = model.critic_head(critic_in).squeeze(-1)
        return logits, value

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)

    lp_all = jax.nn.log_softmax(logits)
    lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)
    ratio = jnp.exp(lp - old_lp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    actor_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * lp_all, axis=-1))
    critic_loss = jnp.mean((values - returns) ** 2)

    return actor_loss - ent_coef * entropy + vf_coef * critic_loss
