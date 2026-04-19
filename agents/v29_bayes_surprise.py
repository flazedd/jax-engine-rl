"""V29: Bayesian-Surprise Shaping — V2 backbone + KL(q_t || q_{t-1}) reward bonus.

Differs from V11:Curiosity: V11 uses entropy reduction H(σ_{t-1}) − H(σ_t),
which only rewards variance shrinkage. V29 uses the full KL between
consecutive Gaussian posteriors, which captures both mean shift (belief
jumps when evidence surprises the agent) and variance change. For diagonal
Gaussians, KL has a closed form:
  KL(q_t || q_{t-1}) = 0.5 * Σ_i [ (σ_ti/σ_{t-1,i})^2 - 1
                                   + ((μ_ti - μ_{t-1,i}) / σ_{t-1,i})^2
                                   - 2 log(σ_ti/σ_{t-1,i}) ]

Architecture is identical to V2:MuOnly; V29 is a training-time modification.
Training uses a dedicated rollout collector that tracks prev (μ, σ) and
adds β_surprise · KL to the reward before advantage computation.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params
from .varibad_hn import varibad_hn_elbo_loss_fn


v29_elbo_loss_fn = varibad_hn_elbo_loss_fn

BETA_SURPRISE = 0.05


class V29BayesSurpriseActorCritic(eqx.Module):
    """V2 architecture; exposes (μ, σ) for surprise computation in rollout."""
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
                 hidden_size=64, latent_dim=4, policy_hidden=16, *, key):
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

        hn_input = hidden_size + latent_dim
        n_params = compute_n_policy_params(obs_size, policy_hidden, n_actions)
        hn = MLP([hn_input, 256, n_params], key=k4)
        self.hyper_net = eqx.tree_at(
            lambda m: m.layers[-1].weight, hn,
            jnp.zeros_like(hn.layers[-1].weight))

        critic_input = hidden_size + 2 * latent_dim + obs_size
        self.critic_head = MLP([critic_input, 64, 1], key=k5)

    def get_mu_sigma(self, hidden):
        post = self.posterior_net(hidden)
        mu = post[:self.latent_dim]
        log_sig = jnp.clip(post[self.latent_dim:], -5.0, 2.0)
        return mu, jnp.exp(log_sig)

    def forward_step(self, aug_input, obs, hidden):
        new_h = self.gru_cell(aug_input, hidden)
        mu, sigma = self.get_mu_sigma(new_h)
        hn_in = jnp.concatenate([new_h, mu])
        policy_params = self.hyper_net(hn_in)
        logits = apply_generated_policy(
            obs, policy_params, self.obs_size, self.policy_hidden, self.n_actions)
        critic_in = jnp.concatenate([new_h, mu, sigma, obs])
        value = self.critic_head(critic_in).squeeze(-1)
        return logits, value, new_h


def v29_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        new_h = model.gru_cell(x, h)
        mu, sigma = model.get_mu_sigma(new_h)
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


def bayesian_surprise_kl(mu_t, sig_t, mu_prev, sig_prev):
    """KL(N(mu_t, sig_t^2) || N(mu_prev, sig_prev^2)) for diagonal Gaussians."""
    var_t = sig_t ** 2
    var_p = sig_prev ** 2
    return 0.5 * jnp.sum(
        (var_t / var_p) - 1.0
        + ((mu_t - mu_prev) ** 2) / var_p
        - 2.0 * (jnp.log(sig_t) - jnp.log(sig_prev)),
        axis=-1)
