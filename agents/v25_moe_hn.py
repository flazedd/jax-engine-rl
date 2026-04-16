"""V25: Mixture-of-Experts HN — K small HyperNetworks + gating.

Instead of one HN generating one policy, K=3 small HNs each generate
a candidate policy. A gating network conditioned on (h, μ) selects
which expert to use (soft mixture over logits). Different experts
specialize in different task clusters, adding capacity without making
any single HN larger.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params
from .varibad_hn import varibad_hn_elbo_loss_fn

v25_moe_elbo_loss_fn = varibad_hn_elbo_loss_fn

N_EXPERTS = 3


class V25MoEHNActorCritic(eqx.Module):
    gru_cell: eqx.nn.GRUCell
    posterior_net: MLP
    decoder_net: MLP
    expert_nets: list          # K HyperNets
    gate_net: MLP              # (h, μ) → K weights
    critic_head: MLP
    hidden_size: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)
    n_experts: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 hidden_size=64, latent_dim=4, policy_hidden=16,
                 n_experts=N_EXPERTS, *, key):
        keys = jax.random.split(key, 5 + n_experts)
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions
        self.n_experts = n_experts

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=keys[0])

        post = MLP([hidden_size, 64, 2 * latent_dim], key=keys[1])
        self.posterior_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), post,
            (jnp.zeros_like(post.layers[-1].weight),
             jnp.zeros_like(post.layers[-1].bias)))

        decoder_input = latent_dim + obs_size + n_actions
        self.decoder_net = MLP([decoder_input, 64, 1 + obs_size], key=keys[2])

        # K expert HyperNets — each generates full policy weights
        hn_input = hidden_size + latent_dim
        n_params = compute_n_policy_params(obs_size, policy_hidden, n_actions)
        experts = []
        for i in range(n_experts):
            hn = MLP([hn_input, 128, n_params], key=keys[3 + i])
            hn = eqx.tree_at(
                lambda m: m.layers[-1].weight, hn,
                jnp.zeros_like(hn.layers[-1].weight))
            experts.append(hn)
        self.expert_nets = experts

        # Gating: (h, μ) → softmax weights over experts
        self.gate_net = MLP([hn_input, 32, n_experts],
                            key=keys[3 + n_experts])

        critic_input = hidden_size + 2 * latent_dim + obs_size
        self.critic_head = MLP([critic_input, 64, 1],
                               key=keys[4 + n_experts])

    def _get_logits(self, new_h, mu, obs):
        hn_in = jnp.concatenate([new_h, mu])

        # Gating weights
        gate_logits = self.gate_net(hn_in)
        gate_weights = jax.nn.softmax(gate_logits)  # (n_experts,)

        # Each expert generates policy logits
        def expert_logits(expert_net):
            pp = expert_net(hn_in)
            return apply_generated_policy(
                obs, pp, self.obs_size, self.policy_hidden, self.n_actions)

        # Stack expert logits and mix
        all_logits = jnp.stack(
            [expert_logits(e) for e in self.expert_nets])  # (K, n_actions)
        mixed_logits = jnp.sum(
            gate_weights[:, None] * all_logits, axis=0)  # (n_actions,)
        return mixed_logits

    def forward_step(self, aug_input, obs, hidden):
        new_h = self.gru_cell(aug_input, hidden)
        post = self.posterior_net(new_h)
        mu = post[:self.latent_dim]
        log_sig = jnp.clip(post[self.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)

        logits = self._get_logits(new_h, mu, obs)

        critic_in = jnp.concatenate([new_h, mu, sigma, obs])
        val = self.critic_head(critic_in).squeeze(-1)
        return logits, val, new_h


def v25_moe_ppo_loss_fn(model, batch,
                        clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        new_h = model.gru_cell(x, h)
        post = model.posterior_net(new_h)
        mu = post[:model.latent_dim]
        log_sig = jnp.clip(post[model.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)
        logits = model._get_logits(new_h, mu, o)
        critic_in = jnp.concatenate([new_h, mu, sigma, o])
        val = model.critic_head(critic_in).squeeze(-1)
        return logits, val

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
