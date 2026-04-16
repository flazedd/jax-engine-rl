"""V9: Normalized HN — generated weights are normalized before use.

Based on V2 (μ-only HN). The HyperNet maps z → W (policy weights).
Without normalization, the magnitude of generated weights is uncontrolled:
small changes in z can produce wildly different weight norms, making the
generated policy's behavior unstable. This manifests as fast early learning
(coarse task discrimination is easy) followed by a stall (fine refinement
requires stable, predictable weight updates).

Fix: normalize the generated weight vector to zero mean and unit variance,
then scale by a learnable parameter. This decouples the direction of
generated weights (which encodes task-specific policy) from their magnitude
(which is learned globally). Similar in spirit to Weight Normalization
(Salimans & Kingma, 2016) applied to HN output.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params
from .varibad_hn import varibad_hn_elbo_loss_fn

v9_normedhn_elbo_loss_fn = varibad_hn_elbo_loss_fn


class V9NormedHNActorCritic(eqx.Module):
    """V2 μ-only + output normalization on generated policy weights.

    After HN generates the weight vector, we normalize it (zero mean,
    unit std) and multiply by a learnable scale factor. This stabilizes
    the z→W mapping's sensitivity throughout training.
    """
    gru_cell: eqx.nn.GRUCell
    posterior_net: MLP
    decoder_net: MLP
    hyper_net: MLP
    critic_head: MLP
    hn_scale: jnp.ndarray      # learnable scalar for weight magnitude
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

        # Learnable scale — init to 1.0, optimizer will tune
        self.hn_scale = jnp.array(1.0)

        critic_input_size = hidden_size + 2 * latent_dim + obs_size
        self.critic_head = MLP([critic_input_size, 64, 1], key=k5)

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def _normalize_params(self, params):
        """Normalize generated weights: zero mean, unit std, then scale."""
        mean = jnp.mean(params)
        std = jnp.std(params) + 1e-8
        return self.hn_scale * (params - mean) / std

    def forward_step(self, aug_input, obs, hidden):
        new_h = self.gru_cell(aug_input, hidden)
        post = self.posterior_net(new_h)
        mu = post[:self.latent_dim]
        log_sig = jnp.clip(post[self.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)
        hn_in = jnp.concatenate([new_h, mu])
        raw_params = self.hyper_net(hn_in)
        policy_params = self._normalize_params(raw_params)
        logits = apply_generated_policy(
            obs, policy_params, self.obs_size, self.policy_hidden, self.n_actions)
        critic_in = jnp.concatenate([new_h, mu, sigma, obs])
        value = self.critic_head(critic_in).squeeze(-1)
        return logits, value, new_h


def v9_normedhn_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01,
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
        raw_params = model.hyper_net(hn_in)
        policy_params = model._normalize_params(raw_params)
        logits = apply_generated_policy(
            o, policy_params, model.obs_size, model.policy_hidden, model.n_actions)
        critic_in = jnp.concatenate([new_h, mu, sigma, o])
        value = model.critic_head(critic_in).squeeze(-1)
        return logits, value

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
