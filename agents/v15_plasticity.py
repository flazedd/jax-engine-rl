"""V15: Plasticity — GRU + modulation of base policy.

Self-modifying network: the base policy has fixed weights learned during
meta-training. The GRU encodes history → h_t. A modulation network maps
h_t to per-neuron gain factors and bias shifts that adapt the base policy
to the inferred task.

MAML meets hypernetworks: the base policy provides a good default,
the modulation provides task-specific adjustment. Unlike full HyperNet
weight generation, the modulation signal is small (2*policy_hidden +
n_actions scalars), making it more stable to train.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits


class V15PlasticityActorCritic(eqx.Module):
    """GRU + modulated base policy actor-critic."""
    gru_cell: eqx.nn.GRUCell
    policy_layer1: eqx.nn.Linear
    policy_layer2: eqx.nn.Linear
    modulation_net: MLP
    critic_head: MLP
    hidden_size: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)
    mod_size: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 hidden_size=64, policy_hidden=16, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.hidden_size = hidden_size
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions
        self.mod_size = 2 * policy_hidden + n_actions

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=k1)
        self.policy_layer1 = eqx.nn.Linear(obs_size, policy_hidden, key=k2)
        self.policy_layer2 = eqx.nn.Linear(policy_hidden, n_actions, key=k3)

        # Modulation net: zero-init last layer → no modulation at start
        mod = MLP([hidden_size, 64, self.mod_size], key=k4)
        self.modulation_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), mod,
            (jnp.zeros_like(mod.layers[-1].weight),
             jnp.zeros_like(mod.layers[-1].bias)))

        self.critic_head = MLP([hidden_size + obs_size, 64, 1], key=k5)

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def forward_step(self, aug_input, obs, hidden):
        new_h = self.gru_cell(aug_input, hidden)

        mod = self.modulation_net(new_h)
        gains = mod[:self.policy_hidden]
        shifts1 = mod[self.policy_hidden:2 * self.policy_hidden]
        shifts2 = mod[2 * self.policy_hidden:]

        h1 = self.policy_layer1(obs) + shifts1
        h1 = jax.nn.relu(h1) * (1.0 + gains)
        logits = self.policy_layer2(h1) + shifts2

        critic_in = jnp.concatenate([new_h, obs])
        val = self.critic_head(critic_in).squeeze(-1)
        return logits, val, new_h


def v15_plasticity_loss_fn(model, batch,
                           clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        new_h = model.gru_cell(x, h)
        mod = model.modulation_net(new_h)
        gains = mod[:model.policy_hidden]
        shifts1 = mod[model.policy_hidden:2 * model.policy_hidden]
        shifts2 = mod[2 * model.policy_hidden:]
        h1 = model.policy_layer1(o) + shifts1
        h1 = jax.nn.relu(h1) * (1.0 + gains)
        logits = model.policy_layer2(h1) + shifts2
        critic_in = jnp.concatenate([new_h, o])
        val = model.critic_head(critic_in).squeeze(-1)
        return logits, val

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
