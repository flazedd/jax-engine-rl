"""V24: Auxiliary Task Classifier — V2 + classification head on μ.

Same architecture as V2 (HN with μ-only conditioning). Adds a small
classification head μ_t → task_logits trained with cross-entropy against
the true task ID. This provides a direct gradient signal that pushes μ
to be maximally discriminative, complementing the indirect ELBO signal.

Requires task IDs during training (passed as 7th element of ELBO data).
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params

CLASSIFY_WEIGHT = 1.0


class V24AuxClassifierActorCritic(eqx.Module):
    gru_cell: eqx.nn.GRUCell
    posterior_net: MLP
    decoder_net: MLP
    classifier: MLP            # μ → task logits
    hyper_net: MLP
    critic_head: MLP
    hidden_size: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)
    n_tasks: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions, n_tasks,
                 hidden_size=64, latent_dim=4, policy_hidden=16, *, key):
        keys = jax.random.split(key, 6)
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions
        self.n_tasks = n_tasks

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=keys[0])

        post = MLP([hidden_size, 64, 2 * latent_dim], key=keys[1])
        self.posterior_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), post,
            (jnp.zeros_like(post.layers[-1].weight),
             jnp.zeros_like(post.layers[-1].bias)))

        decoder_input = latent_dim + obs_size + n_actions
        self.decoder_net = MLP([decoder_input, 64, 1 + obs_size], key=keys[2])

        # Task classifier: μ → task logits
        self.classifier = MLP([latent_dim, 32, n_tasks], key=keys[3])

        hn_input = hidden_size + latent_dim
        n_params = compute_n_policy_params(obs_size, policy_hidden, n_actions)
        hn = MLP([hn_input, 256, n_params], key=keys[4])
        self.hyper_net = eqx.tree_at(
            lambda m: m.layers[-1].weight, hn,
            jnp.zeros_like(hn.layers[-1].weight))

        critic_input = hidden_size + 2 * latent_dim + obs_size
        self.critic_head = MLP([critic_input, 64, 1], key=keys[5])

    def forward_step(self, aug_input, obs, hidden):
        new_h = self.gru_cell(aug_input, hidden)
        post = self.posterior_net(new_h)
        mu = post[:self.latent_dim]
        log_sig = jnp.clip(post[self.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)
        hn_in = jnp.concatenate([new_h, mu])
        policy_params = self.hyper_net(hn_in)
        logits = apply_generated_policy(
            obs, policy_params, self.obs_size, self.policy_hidden, self.n_actions)
        critic_in = jnp.concatenate([new_h, mu, sigma, obs])
        val = self.critic_head(critic_in).squeeze(-1)
        return logits, val, new_h


def v24_aux_elbo_loss_fn(model, elbo_data, rng_key, beta=1.0):
    """ELBO + auxiliary task classification loss.

    elbo_data: 7-tuple — standard 6 + task_ids (n_steps, n_trials)
    """
    obs, actions, rewards, next_obs, prev_act_oh, prev_rew, task_ids = elbo_data
    n_steps, n_trials = obs.shape[:2]
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

        # Classification loss: μ → task
        class_logits = jax.vmap(model.classifier)(mus)  # (n_steps, n_tasks)
        log_probs = jax.nn.log_softmax(class_logits, axis=-1)
        targets = jax.nn.one_hot(tid_s, model.n_tasks)
        class_loss = -jnp.mean(jnp.sum(targets * log_probs, axis=-1))

        return recon_r + recon_o + kl + CLASSIFY_WEIGHT * class_loss

    trial_losses = jax.vmap(forward_trial,
                            in_axes=(1, 1, 1, 1, 1, 1, 1, 1))(
        obs, actions, rewards, next_obs, prev_act_oh, prev_rew, eps, task_ids)
    return beta * jnp.mean(trial_losses)


def v24_aux_ppo_loss_fn(model, batch,
                        clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
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
        val = model.critic_head(critic_in).squeeze(-1)
        return logits, val

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
