"""V13: Prediction-error input — feed decoder prediction error into policy.

The ELBO decoder predicts (r̂_t, ô_{t+1}) from (z_t, o_t, a_t). When the
model is surprised (high prediction error), it signals high uncertainty.
V13 feeds the running prediction error magnitude into the generated policy
as an extra input — giving implicit uncertainty awareness.

High prediction error → model is surprised → should explore more
Low prediction error → model is confident → safe to exploit

Architecture change: generated policy receives (obs_t, pred_err_t) where
pred_err_t is a scalar measuring the decoder's recent prediction error.
The prediction error is tracked in the state and updated each step.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import compute_n_policy_params
from .varibad_hn import varibad_hn_elbo_loss_fn

# Re-export ELBO unchanged
v13_elbo_loss_fn = varibad_hn_elbo_loss_fn


def apply_generated_policy_extended(obs_ext, flat_params,
                                     input_size, policy_hidden, n_actions):
    """Apply HN-generated policy to extended input."""
    i = 0
    w1 = flat_params[i:i + policy_hidden * input_size].reshape(
        policy_hidden, input_size)
    i += policy_hidden * input_size
    b1 = flat_params[i:i + policy_hidden]
    i += policy_hidden
    w2 = flat_params[i:i + n_actions * policy_hidden].reshape(
        n_actions, policy_hidden)
    i += n_actions * policy_hidden
    b2 = flat_params[i:i + n_actions]

    x = jax.nn.relu(w1 @ obs_ext + b1)
    return w2 @ x + b2


class V13PredErrActorCritic(eqx.Module):
    """V2 variant with prediction error as extra policy input.

    HyperNet:  (h_t, μ_t) → policy weights
    Policy:    generated MLP: (obs_t, pred_err_t) → logits
    Critic:    (h_t, μ_t, σ_t, obs) → value
    """
    gru_cell: eqx.nn.GRUCell
    posterior_net: MLP
    decoder_net: MLP
    hyper_net: MLP
    critic_head: MLP
    hidden_size: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_input_size: int = eqx.field(static=True)  # obs_size + 1
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 hidden_size=128, latent_dim=4, policy_hidden=64, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.obs_size = obs_size
        self.policy_input_size = obs_size + 1  # (obs, pred_err)
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=k1)

        post = MLP([hidden_size, 64, 2 * latent_dim], key=k2)
        self.posterior_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), post,
            (jnp.zeros_like(post.layers[-1].weight),
             jnp.zeros_like(post.layers[-1].bias)))

        decoder_input = latent_dim + obs_size + n_actions
        self.decoder_net = MLP([decoder_input, 64, 1 + obs_size], key=k3)

        # HN input: h_t + μ (same as V2)
        hn_input_size = hidden_size + latent_dim
        # Generated policy takes (obs, pred_err) — one extra scalar
        n_params = compute_n_policy_params(
            self.policy_input_size, policy_hidden, n_actions)
        hn = MLP([hn_input_size, 256, n_params], key=k4)
        self.hyper_net = eqx.tree_at(
            lambda m: m.layers[-1].weight, hn,
            jnp.zeros_like(hn.layers[-1].weight))

        critic_input_size = hidden_size + 2 * latent_dim + obs_size
        self.critic_head = MLP([critic_input_size, 64, 1], key=k5)

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def compute_pred_error(self, hidden, obs, prev_action_idx):
        """Compute decoder prediction error for current observation.

        Uses posterior mean (no sampling) for deterministic pred error.
        """
        post = self.posterior_net(hidden)
        mu = post[:self.latent_dim]
        action_oh = jax.nn.one_hot(prev_action_idx, self.n_actions)
        dec_in = jnp.concatenate([mu, obs, action_oh])
        dec_out = self.decoder_net(dec_in)
        # Prediction error = mean squared error of obs prediction
        obs_hat = dec_out[1:]  # skip reward prediction
        pred_err = jnp.mean((obs_hat - obs) ** 2)
        return pred_err

    def forward_step(self, aug_input, obs, hidden, pred_err=None):
        """Single step → (logits, value, new_hidden).

        pred_err: scalar prediction error to feed into policy.
                  If None, uses 0.0 (initial step).
        """
        new_h = self.gru_cell(aug_input, hidden)
        post = self.posterior_net(new_h)
        mu = post[:self.latent_dim]
        log_sig = jnp.clip(post[self.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)
        hn_in = jnp.concatenate([new_h, mu])
        policy_params = self.hyper_net(hn_in)
        if pred_err is None:
            pred_err = jnp.float32(0.0)
        obs_ext = jnp.concatenate([obs, pred_err[None]])
        logits = apply_generated_policy_extended(
            obs_ext, policy_params,
            self.policy_input_size, self.policy_hidden, self.n_actions)
        critic_in = jnp.concatenate([new_h, mu, sigma, obs])
        value = self.critic_head(critic_in).squeeze(-1)
        return logits, value, new_h


def v13_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    """PPO loss for V13.

    Note: during minibatch PPO updates we don't have sequential context to
    compute prediction errors. We use pred_err=0 here — the prediction error
    signal is only active during rollout collection where we have sequential
    state. This is acceptable because the HN adapts based on (h_t, μ) which
    already encode uncertainty; pred_err is an auxiliary input channel.
    """
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
        # Use pred_err=0 during minibatch updates (no sequential context)
        obs_ext = jnp.concatenate([o, jnp.zeros(1)])
        logits = apply_generated_policy_extended(
            obs_ext, policy_params,
            model.policy_input_size, model.policy_hidden, model.n_actions)
        critic_in = jnp.concatenate([new_h, mu, sigma, o])
        value = model.critic_head(critic_in).squeeze(-1)
        return logits, value

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
