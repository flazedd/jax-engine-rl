"""V31: Value-of-Information Critic — V2 + dual exploit/explore critic heads.

Two critic heads:
  V_exploit(h, μ, σ, obs): value-to-go assuming we commit to MAP arm now
  V_explore(h, μ, σ, obs): value-to-go assuming we gather more information

Both heads are trained with MSE against the same GAE return (standard PPO
target). The aggregated value used for GAE is max(V_exploit, V_explore).
This encourages each head to specialize:
  - V_exploit is pulled up by trajectories that committed early and won
  - V_explore is pulled up by trajectories that waited and then won later
The gap between heads provides a value-of-information signal that the
policy can implicitly exploit through the shared GRU+μ features.

Architecture otherwise matches V2:MuOnly. ELBO unchanged.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params
from .varibad_hn import varibad_hn_elbo_loss_fn


v31_elbo_loss_fn = varibad_hn_elbo_loss_fn


class V31VoICriticActorCritic(eqx.Module):
    gru_cell: eqx.nn.GRUCell
    posterior_net: MLP
    decoder_net: MLP
    hyper_net: MLP
    critic_exploit: MLP
    critic_explore: MLP
    hidden_size: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 hidden_size=64, latent_dim=4, policy_hidden=16, *, key):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
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
        self.critic_exploit = MLP([critic_input, 64, 1], key=k5)
        self.critic_explore = MLP([critic_input, 64, 1], key=k6)

    def _forward(self, aug_input, obs, hidden):
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
        v_exp = self.critic_exploit(critic_in).squeeze(-1)
        v_exr = self.critic_explore(critic_in).squeeze(-1)
        return logits, v_exp, v_exr, new_h

    def forward_step(self, aug_input, obs, hidden):
        logits, v_exp, v_exr, new_h = self._forward(aug_input, obs, hidden)
        value = jnp.maximum(v_exp, v_exr)
        return logits, value, new_h


def v31_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        logits, v_exp, v_exr, _ = model._forward(x, o, h)
        return logits, v_exp, v_exr

    logits, v_exp, v_exr = jax.vmap(forward_one)(aug, obs, gru_h)

    # Policy + entropy loss from logits
    lp_all = jax.nn.log_softmax(logits)
    lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)
    ratio = jnp.exp(lp - old_lp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    actor_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * lp_all, axis=-1))

    # Both critics regress to returns
    c_loss_exp = jnp.mean((v_exp - returns) ** 2)
    c_loss_exr = jnp.mean((v_exr - returns) ** 2)
    critic_loss = 0.5 * (c_loss_exp + c_loss_exr)

    return actor_loss - ent_coef * entropy + vf_coef * critic_loss
