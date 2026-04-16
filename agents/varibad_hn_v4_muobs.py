"""V4: μ-as-Obs — posterior belief injected as policy input, not HN input.

HyperNet generates weights from h_t only (like RL²+HN), providing
trajectory-adaptive weights. The generated policy receives [obs, μ_t]
as input, giving it a direct belief state channel. Two conditioning paths:
  - HN(h_t) → adaptive weights (implicit trajectory summary)
  - [obs, μ] → direct belief input (explicit posterior summary)
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params
from .varibad_hn import varibad_hn_elbo_loss_fn

# Re-export ELBO unchanged
v4_muobs_elbo_loss_fn = varibad_hn_elbo_loss_fn


class V4MuObsActorCritic(eqx.Module):
    """VariBAD+HN with μ as policy input instead of HN input.

    HyperNet:  h_t → policy weights              (like RL²+HN)
    Policy:    [obs, μ_t] → logits               (belief as direct input)
    Critic:    [h_t, μ_t, obs] → value
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
    policy_input_size: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 hidden_size=128, latent_dim=4, policy_hidden=64, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions
        self.policy_input_size = obs_size + latent_dim

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=k1)

        post = MLP([hidden_size, 64, 2 * latent_dim], key=k2)
        self.posterior_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), post,
            (jnp.zeros_like(post.layers[-1].weight),
             jnp.zeros_like(post.layers[-1].bias)))

        decoder_input = latent_dim + obs_size + n_actions
        self.decoder_net = MLP([decoder_input, 64, 1 + obs_size], key=k3)

        # HN from h_t only; generates policy for extended input [obs, μ]
        n_params = compute_n_policy_params(
            self.policy_input_size, policy_hidden, n_actions)
        hn = MLP([hidden_size, 256, n_params], key=k4)
        self.hyper_net = eqx.tree_at(
            lambda m: m.layers[-1].weight, hn,
            jnp.zeros_like(hn.layers[-1].weight))

        critic_input_size = hidden_size + latent_dim + obs_size
        self.critic_head = MLP([critic_input_size, 64, 1], key=k5)

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def forward_step(self, aug_input, obs, hidden):
        new_h = self.gru_cell(aug_input, hidden)
        post = self.posterior_net(new_h)
        mu = post[:self.latent_dim]
        policy_params = self.hyper_net(new_h)  # HN from h_t only
        policy_input = jnp.concatenate([obs, mu])  # belief as input
        logits = apply_generated_policy(
            policy_input, policy_params,
            self.policy_input_size, self.policy_hidden, self.n_actions)
        critic_in = jnp.concatenate([new_h, mu, obs])
        value = self.critic_head(critic_in).squeeze(-1)
        return logits, value, new_h


def v4_muobs_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        new_h = model.gru_cell(x, h)
        post = model.posterior_net(new_h)
        mu = post[:model.latent_dim]
        policy_params = model.hyper_net(new_h)
        policy_input = jnp.concatenate([o, mu])
        logits = apply_generated_policy(
            policy_input, policy_params,
            model.policy_input_size, model.policy_hidden, model.n_actions)
        critic_in = jnp.concatenate([new_h, mu, o])
        value = model.critic_head(critic_in).squeeze(-1)
        return logits, value

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
