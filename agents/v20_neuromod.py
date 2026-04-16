"""V20: Neuromodulation — context-dependent activation functions.

Instead of the hypernetwork generating weights, it generates the activation
function parameters of the policy network. Each neuron gets a task-specific
PReLU slope: neurons that are "on" in one regime can be "off" in another.

Biologically inspired: neuromodulators (dopamine, serotonin) don't rewrite
synaptic weights — they modulate how neurons respond to inputs. The GRU is
the neuromodulatory system, the policy is the cortex.

Much smaller intervention than full weight generation: one scalar per
neuron (policy_hidden) instead of a full weight matrix. Trains faster
and is more stable.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits


class V20NeuromodActorCritic(eqx.Module):
    """GRU + neuromodulated base policy actor-critic."""
    gru_cell: eqx.nn.GRUCell
    policy_layer1: eqx.nn.Linear
    policy_layer2: eqx.nn.Linear
    modulator_net: MLP         # h_t → PReLU slopes (policy_hidden,)
    critic_head: MLP
    hidden_size: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 hidden_size=64, policy_hidden=16, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.hidden_size = hidden_size
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=k1)
        self.policy_layer1 = eqx.nn.Linear(obs_size, policy_hidden, key=k2)
        self.policy_layer2 = eqx.nn.Linear(policy_hidden, n_actions, key=k3)

        # Modulator: zero-init → slopes start at 0 (like ReLU, since
        # PReLU with slope=0 is ReLU). This gives stable init.
        mod = MLP([hidden_size, 32, policy_hidden], key=k4)
        self.modulator_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), mod,
            (jnp.zeros_like(mod.layers[-1].weight),
             jnp.zeros_like(mod.layers[-1].bias)))

        self.critic_head = MLP([hidden_size + obs_size, 64, 1], key=k5)

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def forward_step(self, aug_input, obs, hidden):
        new_h = self.gru_cell(aug_input, hidden)

        # Task-specific PReLU slopes
        slopes = self.modulator_net(new_h)

        h1 = self.policy_layer1(obs)
        h1 = jnp.where(h1 > 0, h1, slopes * h1)   # PReLU
        logits = self.policy_layer2(h1)

        critic_in = jnp.concatenate([new_h, obs])
        val = self.critic_head(critic_in).squeeze(-1)
        return logits, val, new_h


def v20_neuromod_loss_fn(model, batch,
                         clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        new_h = model.gru_cell(x, h)
        slopes = model.modulator_net(new_h)
        h1 = model.policy_layer1(o)
        h1 = jnp.where(h1 > 0, h1, slopes * h1)
        logits = model.policy_layer2(h1)
        critic_in = jnp.concatenate([new_h, o])
        val = model.critic_head(critic_in).squeeze(-1)
        return logits, val

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
