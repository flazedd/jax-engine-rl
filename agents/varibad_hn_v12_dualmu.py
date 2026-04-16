"""V12: Dual μ input — feed (obs_t, μ_t) into the HN-generated policy.

V2 feeds only obs_t into the generated policy, with task belief flowing
indirectly through the HN weights. This gives dual conditioning but
the "belief path" only reshapes the function — it can't directly shift
the policy based on the current belief point.

V12 feeds (obs_t, μ_t) into the generated policy. Now task belief flows
both through the HN weights (function shape) AND as a direct input
(operating point). The generated policy can learn "if μ points to task 3
and obs is X, then do Y" — more expressive than either path alone.

Key change: generated policy input size = obs_size + latent_dim.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import compute_n_policy_params
from .varibad_hn import varibad_hn_elbo_loss_fn

# Re-export ELBO unchanged
v12_elbo_loss_fn = varibad_hn_elbo_loss_fn


def apply_generated_policy_extended(obs_mu, flat_params,
                                     input_size, policy_hidden, n_actions):
    """Apply HN-generated policy to extended input (obs, μ).

    Same structure as apply_generated_policy but with larger input dim.
    """
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

    x = jax.nn.relu(w1 @ obs_mu + b1)
    return w2 @ x + b2


class V12DualMuActorCritic(eqx.Module):
    """V2 variant with (obs, μ) as generated policy input.

    HyperNet:  (h_t, μ_t) → policy weights
    Policy:    generated MLP: (obs_t, μ_t) → logits  (dual mu conditioning)
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
    policy_input_size: int = eqx.field(static=True)  # obs_size + latent_dim
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 hidden_size=128, latent_dim=4, policy_hidden=64, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.obs_size = obs_size
        self.policy_input_size = obs_size + latent_dim  # (obs, μ)
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
        # Generated policy takes (obs, μ) — larger input
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

    def forward_step(self, aug_input, obs, hidden):
        new_h = self.gru_cell(aug_input, hidden)
        post = self.posterior_net(new_h)
        mu = post[:self.latent_dim]
        log_sig = jnp.clip(post[self.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)
        hn_in = jnp.concatenate([new_h, mu])
        policy_params = self.hyper_net(hn_in)
        # Generated policy receives (obs, μ)
        obs_mu = jnp.concatenate([obs, mu])
        logits = apply_generated_policy_extended(
            obs_mu, policy_params,
            self.policy_input_size, self.policy_hidden, self.n_actions)
        critic_in = jnp.concatenate([new_h, mu, sigma, obs])
        value = self.critic_head(critic_in).squeeze(-1)
        return logits, value, new_h


def v12_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
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
        obs_mu = jnp.concatenate([o, mu])
        logits = apply_generated_policy_extended(
            obs_mu, policy_params,
            model.policy_input_size, model.policy_hidden, model.n_actions)
        critic_in = jnp.concatenate([new_h, mu, sigma, o])
        value = model.critic_head(critic_in).squeeze(-1)
        return logits, value

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
