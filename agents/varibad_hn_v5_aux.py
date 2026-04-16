"""V5: Auxiliary Regime Prediction — semi-supervised signal on posterior.

Adds a small head μ → task_logits trained alongside the ELBO via
cross-entropy on the true task/regime label. Pushes μ toward a
regime-structured representation during training. Not used at test time.

The training script must collect task_ids from the env state and pass
them as the last element of elbo_data.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params
from .varibad_hn import varibad_hn_ppo_loss_fn

# PPO loss is identical to base VariBAD+HN (regime_head not used in policy)
v5_aux_ppo_loss_fn = varibad_hn_ppo_loss_fn


class V5AuxActorCritic(eqx.Module):
    """VariBAD+HN with auxiliary regime prediction head.

    Same as VariBADHNActorCritic plus:
    regime_head: μ_t → task logits (n_tasks classes)
    """
    gru_cell: eqx.nn.GRUCell
    posterior_net: MLP
    decoder_net: MLP
    hyper_net: MLP
    critic_head: MLP
    regime_head: MLP
    hidden_size: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)
    n_tasks: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions, n_tasks,
                 hidden_size=128, latent_dim=4, policy_hidden=64, *, key):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions
        self.n_tasks = n_tasks

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=k1)

        post = MLP([hidden_size, 64, 2 * latent_dim], key=k2)
        self.posterior_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), post,
            (jnp.zeros_like(post.layers[-1].weight),
             jnp.zeros_like(post.layers[-1].bias)))

        decoder_input = latent_dim + obs_size + n_actions
        self.decoder_net = MLP([decoder_input, 64, 1 + obs_size], key=k3)

        hn_input_size = hidden_size + 2 * latent_dim
        n_params = compute_n_policy_params(obs_size, policy_hidden, n_actions)
        hn = MLP([hn_input_size, 256, n_params], key=k4)
        self.hyper_net = eqx.tree_at(
            lambda m: m.layers[-1].weight, hn,
            jnp.zeros_like(hn.layers[-1].weight))

        critic_input_size = hidden_size + 2 * latent_dim + obs_size
        self.critic_head = MLP([critic_input_size, 64, 1], key=k5)

        # Auxiliary: μ → task logits
        self.regime_head = MLP([latent_dim, 32, n_tasks], key=k6)

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def _get_posterior(self, h):
        post = self.posterior_net(h)
        mu = post[:self.latent_dim]
        log_sig = jnp.clip(post[self.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)
        return mu, sigma

    def forward_step(self, aug_input, obs, hidden):
        new_h = self.gru_cell(aug_input, hidden)
        mu, sigma = self._get_posterior(new_h)
        hn_in = jnp.concatenate([new_h, mu, sigma])
        policy_params = self.hyper_net(hn_in)
        logits = apply_generated_policy(
            obs, policy_params, self.obs_size, self.policy_hidden, self.n_actions)
        critic_in = jnp.concatenate([new_h, mu, sigma, obs])
        value = self.critic_head(critic_in).squeeze(-1)
        return logits, value, new_h


def v5_aux_elbo_loss_fn(model, elbo_data, rng_key, beta=1.0, aux_weight=1.0):
    """ELBO + auxiliary regime prediction loss.

    elbo_data: (obs, actions, rewards, next_obs, prev_act_oh, prev_rew, task_ids)
               task_ids: (n_steps, n_trials) int32
    """
    obs, actions, rewards, next_obs, prev_act_oh, prev_rew, task_ids = elbo_data

    n_steps = obs.shape[0]
    n_trials = obs.shape[1]
    eps = jax.random.normal(rng_key, (n_steps, n_trials, model.latent_dim))

    def forward_trial(obs_s, act_s, rew_s, nobs_s, pa_s, pr_s, eps_s, tid_s):
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

        # Standard ELBO
        act_oh = jax.nn.one_hot(act_s, model.n_actions)
        dec_in = jnp.concatenate([zs, obs_s, act_oh], axis=-1)
        dec_out = jax.vmap(model.decoder_net)(dec_in)
        r_hat = dec_out[:, 0]
        o_hat = dec_out[:, 1:]
        recon_r = jnp.mean((r_hat - rew_s) ** 2)
        recon_o = jnp.mean((o_hat - nobs_s) ** 2)
        kl = 0.5 * jnp.mean(jnp.sum(
            mus ** 2 + jnp.exp(2 * log_sigs) - 1 - 2 * log_sigs, axis=-1))

        # Auxiliary: predict task from μ
        regime_logits = jax.vmap(model.regime_head)(mus)  # (T, n_tasks)
        task_oh = jax.nn.one_hot(tid_s, model.n_tasks)
        aux_loss = -jnp.mean(jnp.sum(
            task_oh * jax.nn.log_softmax(regime_logits), axis=-1))

        return recon_r + recon_o + kl + aux_weight * aux_loss

    trial_losses = jax.vmap(
        forward_trial, in_axes=(1, 1, 1, 1, 1, 1, 1, 1))(
        obs, actions, rewards, next_obs, prev_act_oh, prev_rew, eps, task_ids)
    return beta * jnp.mean(trial_losses)
