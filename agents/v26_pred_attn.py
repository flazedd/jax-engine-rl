"""V26: Predictive Attention — V16 + V18 + V2 triple fusion.

Attention buffer for selective memory (V16), predictive coding ELBO
to force informative latents (V18), HN with μ conditioning for
expressive policy generation (V2). Everything reinforces everything:
attention retrieves relevant past data, posterior encodes task belief,
predictive loss shapes the latent to be useful, HN generates policy.

Hidden state = [gru_h | buffer_flat | write_idx]
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params


class V26PredAttnActorCritic(eqx.Module):
    gru_cell: eqx.nn.GRUCell
    posterior_net: MLP
    decoder_net: MLP           # predictive: (z, action_oh) → (r, next_obs)
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    query_proj: eqx.nn.Linear
    hyper_net: MLP
    critic_head: MLP
    gru_hidden_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    buffer_size: int = eqx.field(static=True)
    entry_size: int = eqx.field(static=True)
    attn_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 gru_hidden=64, buffer_size=10, attn_dim=16,
                 latent_dim=4, policy_hidden=16, *, key):
        keys = jax.random.split(key, 9)
        self.gru_hidden_size = gru_hidden
        self.buffer_size = buffer_size
        self.entry_size = input_size
        self.attn_dim = attn_dim
        self.latent_dim = latent_dim
        self.hidden_size = gru_hidden + buffer_size * input_size + 1
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions

        self.gru_cell = eqx.nn.GRUCell(input_size, gru_hidden, key=keys[0])

        post = MLP([gru_hidden, 64, 2 * latent_dim], key=keys[1])
        self.posterior_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), post,
            (jnp.zeros_like(post.layers[-1].weight),
             jnp.zeros_like(post.layers[-1].bias)))

        # Predictive decoder: (z, action) → (r, next_obs) — no obs
        decoder_input = latent_dim + n_actions
        self.decoder_net = MLP([decoder_input, 64, 1 + obs_size], key=keys[2])

        self.key_proj = eqx.nn.Linear(input_size, attn_dim, key=keys[3])
        self.value_proj = eqx.nn.Linear(input_size, attn_dim, key=keys[4])
        self.query_proj = eqx.nn.Linear(obs_size, attn_dim, key=keys[5])

        # HN: (attention_context, μ) → policy
        hn_input = attn_dim + latent_dim
        n_params = compute_n_policy_params(obs_size, policy_hidden, n_actions)
        hn = MLP([hn_input, 256, n_params], key=keys[6])
        self.hyper_net = eqx.tree_at(
            lambda m: m.layers[-1].weight, hn,
            jnp.zeros_like(hn.layers[-1].weight))

        critic_input = gru_hidden + attn_dim + 2 * latent_dim + obs_size
        self.critic_head = MLP([critic_input, 64, 1], key=keys[7])

    def _forward(self, aug_input, obs, hidden):
        gru_h = hidden[:self.gru_hidden_size]
        buf_idx = hidden[self.gru_hidden_size:]
        buf = buf_idx[:-1].reshape(self.buffer_size, self.entry_size)
        write_idx = jnp.int32(jnp.round(buf_idx[-1]))

        new_h = self.gru_cell(aug_input, gru_h)
        post = self.posterior_net(new_h)
        mu = post[:self.latent_dim]
        log_sig = jnp.clip(post[self.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)

        # Attention
        pos = write_idx % self.buffer_size
        buf = buf.at[pos].set(aug_input)
        new_idx = write_idx + 1
        n_valid = jnp.minimum(new_idx, self.buffer_size)

        query = self.query_proj(obs)
        ks = jax.vmap(self.key_proj)(buf)
        vs = jax.vmap(self.value_proj)(buf)
        scores = ks @ query / jnp.sqrt(jnp.float32(self.attn_dim))
        mask = jnp.arange(self.buffer_size) < n_valid
        scores = jnp.where(mask, scores, -1e9)
        context = jax.nn.softmax(scores) @ vs

        # HN from (attention_context, μ)
        hn_in = jnp.concatenate([context, mu])
        policy_params = self.hyper_net(hn_in)
        logits = apply_generated_policy(
            obs, policy_params, self.obs_size, self.policy_hidden, self.n_actions)

        critic_in = jnp.concatenate([new_h, context, mu, sigma, obs])
        val = self.critic_head(critic_in).squeeze(-1)

        new_hidden = jnp.concatenate([new_h, buf.ravel(),
                                       jnp.float32(new_idx)[None]])
        return logits, val, new_hidden

    def forward_step(self, aug_input, obs, hidden):
        return self._forward(aug_input, obs, hidden)


def v26_pred_attn_elbo_loss_fn(model, elbo_data, rng_key, beta=1.0):
    """Predictive ELBO (no obs in decoder) using gru_hidden_size."""
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
        dec_in = jnp.concatenate([zs, act_oh], axis=-1)  # no obs
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


def v26_pred_attn_ppo_loss_fn(model, batch,
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
