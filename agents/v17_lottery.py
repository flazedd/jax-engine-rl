"""V17: Lottery Ticket — binary mask over large random policy network.

From the lottery ticket hypothesis (Frankle & Carlin 2019): a large random
network contains many good subnetworks. GRU encodes history → h_t, a mask
network generates binary masks (via sigmoid + STE) that select a
task-specific subnetwork from a fixed random policy.

Each regime activates a different "circuit." No weight generation, no
gradient scaling issues. The mask net output is just sigmoid logits —
much simpler than generating hundreds of continuous weights.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits


class V17LotteryActorCritic(eqx.Module):
    """GRU + lottery-ticket masked policy actor-critic."""
    gru_cell: eqx.nn.GRUCell
    random_w1: jnp.ndarray
    random_b1: jnp.ndarray
    random_w2: jnp.ndarray
    random_b2: jnp.ndarray
    mask_net: MLP
    critic_head: MLP
    hidden_size: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    lottery_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)
    n_mask_params: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 hidden_size=64, lottery_hidden=64, *, key):
        keys = jax.random.split(key, 7)
        self.hidden_size = hidden_size
        self.obs_size = obs_size
        self.lottery_hidden = lottery_hidden
        self.n_actions = n_actions

        self.n_mask_params = (lottery_hidden * obs_size + lottery_hidden +
                              n_actions * lottery_hidden + n_actions)

        # Fixed random weights — frozen via stop_gradient in forward pass
        self.random_w1 = jax.random.normal(
            keys[1], (lottery_hidden, obs_size)) * 0.1
        self.random_b1 = jax.random.normal(
            keys[2], (lottery_hidden,)) * 0.1
        self.random_w2 = jax.random.normal(
            keys[3], (n_actions, lottery_hidden)) * 0.1
        self.random_b2 = jax.random.normal(
            keys[4], (n_actions,)) * 0.1

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=keys[0])

        # Mask net: zero-init last layer → sigmoid(0) = 0.5 at start
        mask = MLP([hidden_size, 128, self.n_mask_params], key=keys[5])
        self.mask_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), mask,
            (jnp.zeros_like(mask.layers[-1].weight),
             jnp.zeros_like(mask.layers[-1].bias)))

        self.critic_head = MLP([hidden_size + obs_size, 64, 1], key=keys[6])

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def _apply_masked_policy(self, h, obs):
        """Generate mask from h, apply to frozen random weights."""
        mask_logits = self.mask_net(h)
        mask_soft = jax.nn.sigmoid(mask_logits)
        mask_hard = (mask_soft > 0.5).astype(jnp.float32)
        mask = mask_hard - jax.lax.stop_gradient(mask_soft) + mask_soft

        # Frozen random weights
        w1 = jax.lax.stop_gradient(self.random_w1)
        b1 = jax.lax.stop_gradient(self.random_b1)
        w2 = jax.lax.stop_gradient(self.random_w2)
        b2 = jax.lax.stop_gradient(self.random_b2)

        # Unpack mask
        lh, os, na = self.lottery_hidden, self.obs_size, self.n_actions
        i = 0
        m_w1 = mask[i:i + lh * os].reshape(lh, os);  i += lh * os
        m_b1 = mask[i:i + lh];                        i += lh
        m_w2 = mask[i:i + na * lh].reshape(na, lh);   i += na * lh
        m_b2 = mask[i:i + na]

        h1 = jax.nn.relu((m_w1 * w1) @ obs + m_b1 * b1)
        return (m_w2 * w2) @ h1 + m_b2 * b2

    def forward_step(self, aug_input, obs, hidden):
        new_h = self.gru_cell(aug_input, hidden)
        logits = self._apply_masked_policy(new_h, obs)
        critic_in = jnp.concatenate([new_h, obs])
        val = self.critic_head(critic_in).squeeze(-1)
        return logits, val, new_h


def v17_lottery_loss_fn(model, batch,
                        clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        new_h = model.gru_cell(x, h)
        logits = model._apply_masked_policy(new_h, o)
        critic_in = jnp.concatenate([new_h, o])
        val = model.critic_head(critic_in).squeeze(-1)
        return logits, val

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
