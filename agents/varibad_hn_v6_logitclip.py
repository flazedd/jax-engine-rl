"""V6: Logit Clip — bounds generated policy logits to prevent entropy collapse.

Based on V2 (μ-only HN). The HyperNet can produce arbitrarily large logits,
causing the policy to become near-deterministic early in training. Once
exploration dies, the policy locks into a suboptimal greedy solution and
cannot refine further.

Fix: clip generated logits to [-L, L] via L·tanh(logits/L). This is a
soft bound that preserves gradients while preventing entropy collapse.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params
from .varibad_hn import varibad_hn_elbo_loss_fn

v6_logitclip_elbo_loss_fn = varibad_hn_elbo_loss_fn

LOGIT_SCALE = 3.0  # soft bound on logit magnitude


class V6LogitClipActorCritic(eqx.Module):
    """V2 μ-only + soft logit clipping on generated policy.

    HyperNet:  (h_t, μ_t) → policy weights
    Policy:    obs → raw logits → L·tanh(logits/L)
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

        hn_input_size = hidden_size + latent_dim
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
        raw_logits = apply_generated_policy(
            obs, policy_params, self.obs_size, self.policy_hidden, self.n_actions)
        logits = LOGIT_SCALE * jnp.tanh(raw_logits / LOGIT_SCALE)
        critic_in = jnp.concatenate([new_h, mu, sigma, obs])
        value = self.critic_head(critic_in).squeeze(-1)
        return logits, value, new_h


def v6_logitclip_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01,
                              vf_coef=0.5):
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
        raw_logits = apply_generated_policy(
            o, policy_params, model.obs_size, model.policy_hidden, model.n_actions)
        logits = LOGIT_SCALE * jnp.tanh(raw_logits / LOGIT_SCALE)
        critic_in = jnp.concatenate([new_h, mu, sigma, o])
        value = model.critic_head(critic_in).squeeze(-1)
        return logits, value

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
