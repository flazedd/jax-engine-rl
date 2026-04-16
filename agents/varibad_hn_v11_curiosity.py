"""V11: Curiosity bonus — intrinsic reward for posterior entropy reduction.

Adds r_intrinsic = β_curiosity · (H(σ_t) - H(σ_{t+1})) to the reward.
Actions that reduce posterior uncertainty (informative actions) get a bonus.
This directly incentivizes strategic exploration without the agent having
to discover it purely through RL reward signal.

The bonus decays naturally as the posterior sharpens — early exploration
is rewarded more than late exploration (when the posterior is already tight).

Architecture is identical to V2:MuOnly. The change is in the rollout
collection where intrinsic reward is added to extrinsic reward.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params
from .varibad_hn import varibad_hn_elbo_loss_fn

# Re-export ELBO unchanged
v11_elbo_loss_fn = varibad_hn_elbo_loss_fn

# Curiosity bonus coefficient
BETA_CURIOSITY = 0.1


class V11CuriosityActorCritic(eqx.Module):
    """V2:MuOnly architecture with curiosity bonus support.

    Same as V2 but forward_step also returns sigma for entropy computation.
    HyperNet:  (h_t, μ_t) → policy weights
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

        post = MLP([hidden_size, 64, 2 * latent_dim], key=k2)
        self.posterior_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), post,
            (jnp.zeros_like(post.layers[-1].weight),
             jnp.zeros_like(post.layers[-1].bias)))

        decoder_input = latent_dim + obs_size + n_actions
        self.decoder_net = MLP([decoder_input, 64, 1 + obs_size], key=k3)

        hn_input_size = hidden_size + latent_dim  # μ only
        n_params = compute_n_policy_params(obs_size, policy_hidden, n_actions)
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
        logits = apply_generated_policy(
            obs, policy_params, self.obs_size, self.policy_hidden, self.n_actions)
        critic_in = jnp.concatenate([new_h, mu, sigma, obs])
        value = self.critic_head(critic_in).squeeze(-1)
        return logits, value, new_h

    def get_log_sigma(self, hidden):
        """Extract log_sigma from hidden state (for entropy computation)."""
        post = self.posterior_net(hidden)
        log_sig = jnp.clip(post[self.latent_dim:], -5.0, 2.0)
        return log_sig


def posterior_entropy(log_sigma):
    """Differential entropy of diagonal Gaussian: 0.5 * sum(log(2πe σ²))."""
    return 0.5 * jnp.sum(2.0 * log_sigma + jnp.log(2.0 * jnp.pi * jnp.e))


def v11_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
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
