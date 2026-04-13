"""RL²+HN — GRU with HyperNetwork-generated policy (Beck et al. 2023)."""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP


def apply_generated_policy(obs, flat_params, obs_size, policy_hidden, n_actions):
    """Apply HyperNet-generated policy MLP to observation.

    Generated MLP: obs → Linear(obs_size, policy_hidden) → ReLU
                        → Linear(policy_hidden, n_actions) → logits

    flat_params: (n_policy_params,)
    """
    i = 0
    w1 = flat_params[i:i + policy_hidden * obs_size].reshape(
        policy_hidden, obs_size)
    i += policy_hidden * obs_size
    b1 = flat_params[i:i + policy_hidden]
    i += policy_hidden
    w2 = flat_params[i:i + n_actions * policy_hidden].reshape(
        n_actions, policy_hidden)
    i += n_actions * policy_hidden
    b2 = flat_params[i:i + n_actions]

    x = jax.nn.relu(w1 @ obs + b1)
    return w2 @ x + b2


def compute_n_policy_params(obs_size, policy_hidden, n_actions):
    """Total flat parameters for the generated policy MLP."""
    return (obs_size * policy_hidden + policy_hidden +
            policy_hidden * n_actions + n_actions)


class HNActorCritic(eqx.Module):
    """GRU + HyperNetwork actor-critic.

    GRU processes (obs, prev_action_onehot, prev_reward).
    HyperNet: gru_hidden → flat policy MLP weights.
    Generated policy: obs → logits (dual conditioning).
    Critic: [gru_hidden, obs] → value (skip connection).
    """
    gru_cell: eqx.nn.GRUCell
    hyper_net: MLP
    critic_head: MLP
    hidden_size: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 hidden_size=128, policy_hidden=64, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.hidden_size = hidden_size
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions
        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=k1)

        n_params = compute_n_policy_params(obs_size, policy_hidden, n_actions)
        hn = MLP([hidden_size, 256, n_params], key=k2)
        # Bias-HyperInit: zero last-layer weights so all h_t produce
        # the same base policy at initialization (bias is kept random)
        self.hyper_net = eqx.tree_at(
            lambda m: m.layers[-1].weight, hn,
            jnp.zeros_like(hn.layers[-1].weight))

        self.critic_head = MLP([hidden_size + obs_size, 64, 1], key=k3)

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def forward_step(self, aug_input, obs, hidden):
        """Single step → (logits, value, new_hidden)."""
        new_h = self.gru_cell(aug_input, hidden)
        policy_params = self.hyper_net(new_h)
        logits = apply_generated_policy(
            obs, policy_params, self.obs_size, self.policy_hidden, self.n_actions)
        critic_in = jnp.concatenate([new_h, obs])
        value = self.critic_head(critic_in).squeeze(-1)
        return logits, value, new_h


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def rl2_hn_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    """Combined PPO loss for RL²+HN.

    batch: (obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns)
    """
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        new_h = model.gru_cell(x, h)
        policy_params = model.hyper_net(new_h)
        logits = apply_generated_policy(
            o, policy_params, model.obs_size, model.policy_hidden, model.n_actions)
        critic_in = jnp.concatenate([new_h, o])
        value = model.critic_head(critic_in).squeeze(-1)
        return logits, value

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)

    lp_all = jax.nn.log_softmax(logits)
    lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)
    ratio = jnp.exp(lp - old_lp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    actor_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * lp_all, axis=-1))
    critic_loss = jnp.mean((values - returns) ** 2)

    return actor_loss - ent_coef * entropy + vf_coef * critic_loss
