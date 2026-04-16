"""V19: World Model — counterfactual rollouts for action selection.

GRU encoder + small world model (transition + reward predictor). At each
step, for every possible action, the world model predicts the immediate
reward. These predicted rewards become additional input to the policy,
giving it a model-based planning signal.

This sidesteps the belief-to-action mapping problem: instead of learning
which action is best from the hidden state directly, the agent can
simulate outcomes and pick the best-looking action.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits


class V19WorldModelActorCritic(eqx.Module):
    """GRU + world model with counterfactual predictions."""
    gru_cell: eqx.nn.GRUCell
    world_model: MLP           # (h, obs, action_oh) → (reward, next_obs)
    actor_head: MLP            # (obs, predicted_rewards) → logits
    critic_head: MLP           # (h, obs) → value
    hidden_size: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 hidden_size=64, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.hidden_size = hidden_size
        self.obs_size = obs_size
        self.n_actions = n_actions

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=k1)

        wm_input = hidden_size + obs_size + n_actions
        wm_output = 1 + obs_size    # reward + next_obs
        self.world_model = MLP([wm_input, 64, wm_output], key=k2)

        # Policy receives obs + predicted reward per action
        self.actor_head = MLP([obs_size + n_actions, 64, n_actions], key=k3)
        self.critic_head = MLP([hidden_size + obs_size, 64, 1], key=k4)

    def init_state(self):
        return jnp.zeros(self.hidden_size)

    def forward_step(self, aug_input, obs, hidden):
        new_h = self.gru_cell(aug_input, hidden)

        # Predict reward for each action (counterfactual evaluation)
        def predict_reward(a_idx):
            a_oh = jax.nn.one_hot(a_idx, self.n_actions)
            wm_in = jnp.concatenate([new_h, obs, a_oh])
            return self.world_model(wm_in)[0]

        pred_rewards = jax.vmap(predict_reward)(jnp.arange(self.n_actions))

        policy_in = jnp.concatenate([obs, pred_rewards])
        logits = self.actor_head(policy_in)

        critic_in = jnp.concatenate([new_h, obs])
        val = self.critic_head(critic_in).squeeze(-1)
        return logits, val, new_h


# ---------------------------------------------------------------------------
# World model training loss (prediction, no VAE)
# ---------------------------------------------------------------------------

def v19_wm_loss_fn(model, elbo_data, rng_key, beta=1.0):
    """Train world model via prediction: (h, obs, action) → (reward, next_obs)."""
    obs, actions, rewards, next_obs, prev_act_oh, prev_rew = elbo_data
    n_steps = obs.shape[0]
    n_trials = obs.shape[1]

    def forward_trial(obs_s, act_s, rew_s, nobs_s, pa_s, pr_s):
        def scan_fn(h, inp):
            o, pa, pr = inp
            aug = jnp.concatenate([o, pa, pr])
            h = model.gru_cell(aug, h)
            return h, h

        h0 = jnp.zeros(model.hidden_size)
        _, hs = jax.lax.scan(scan_fn, h0, (obs_s, pa_s, pr_s))

        act_oh = jax.nn.one_hot(act_s, model.n_actions)
        wm_in = jnp.concatenate([hs, obs_s, act_oh], axis=-1)
        wm_out = jax.vmap(model.world_model)(wm_in)

        r_hat = wm_out[:, 0]
        o_hat = wm_out[:, 1:]
        pred_r = jnp.mean((r_hat - rew_s) ** 2)
        pred_o = jnp.mean((o_hat - nobs_s) ** 2)
        return pred_r + pred_o

    trial_losses = jax.vmap(forward_trial, in_axes=(1, 1, 1, 1, 1, 1))(
        obs, actions, rewards, next_obs, prev_act_oh, prev_rew)
    return beta * jnp.mean(trial_losses)


def v19_wm_ppo_loss_fn(model, batch,
                       clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        new_h = model.gru_cell(x, h)

        def predict_reward(a_idx):
            a_oh = jax.nn.one_hot(a_idx, model.n_actions)
            wm_in = jnp.concatenate([new_h, o, a_oh])
            return model.world_model(wm_in)[0]

        pred_rewards = jax.vmap(predict_reward)(jnp.arange(model.n_actions))
        policy_in = jnp.concatenate([o, pred_rewards])
        logits = model.actor_head(policy_in)
        critic_in = jnp.concatenate([new_h, o])
        val = model.critic_head(critic_in).squeeze(-1)
        return logits, val

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
