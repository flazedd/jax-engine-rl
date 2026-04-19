"""V28: Discrete-Posterior HN — softmax posterior over K tasks + HN.

Replaces V2's continuous Gaussian latent with a discrete distribution over
the K known tasks. The posterior head outputs K logits; belief = softmax(logits).
HyperNet conditions on (h, belief) so policy weights are a function of the
current K-simplex posterior — matches the 1-of-K structure of the env.

Losses combine:
  - CE(belief, true_task_id)        — strong supervised structural signal
  - reconstruction(r, o' | belief, o, a)   — decoder reads belief as input
  - KL(belief || uniform)           — mild entropy regularizer

Uses task_ids during training (via train_recurrent_taskids).
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params


CLASSIFY_WEIGHT = 1.0
RECON_WEIGHT = 0.5
KL_UNIF_WEIGHT = 0.01


class V28DiscretePosteriorActorCritic(eqx.Module):
    gru_cell: eqx.nn.GRUCell
    posterior_net: MLP           # h → K logits over tasks
    decoder_net: MLP             # (belief, obs, action_oh) → (r, next_obs)
    hyper_net: MLP               # (h, belief) → flat policy MLP params
    critic_head: MLP             # (h, belief, obs) → value
    hidden_size: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)
    n_tasks: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions, n_tasks,
                 hidden_size=64, policy_hidden=16, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.hidden_size = hidden_size
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions
        self.n_tasks = n_tasks

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=k1)

        post = MLP([hidden_size, 64, n_tasks], key=k2)
        self.posterior_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), post,
            (jnp.zeros_like(post.layers[-1].weight),
             jnp.zeros_like(post.layers[-1].bias)))

        decoder_input = n_tasks + obs_size + n_actions
        self.decoder_net = MLP([decoder_input, 64, 1 + obs_size], key=k3)

        hn_input = hidden_size + n_tasks
        n_params = compute_n_policy_params(obs_size, policy_hidden, n_actions)
        hn = MLP([hn_input, 256, n_params], key=k4)
        self.hyper_net = eqx.tree_at(
            lambda m: m.layers[-1].weight, hn,
            jnp.zeros_like(hn.layers[-1].weight))

        critic_input = hidden_size + n_tasks + obs_size
        self.critic_head = MLP([critic_input, 64, 1], key=k5)

    def forward_step(self, aug_input, obs, hidden):
        new_h = self.gru_cell(aug_input, hidden)
        logits_post = self.posterior_net(new_h)
        belief = jax.nn.softmax(logits_post)
        hn_in = jnp.concatenate([new_h, belief])
        policy_params = self.hyper_net(hn_in)
        logits = apply_generated_policy(
            obs, policy_params, self.obs_size, self.policy_hidden, self.n_actions)
        critic_in = jnp.concatenate([new_h, belief, obs])
        value = self.critic_head(critic_in).squeeze(-1)
        return logits, value, new_h


def v28_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        new_h = model.gru_cell(x, h)
        belief = jax.nn.softmax(model.posterior_net(new_h))
        hn_in = jnp.concatenate([new_h, belief])
        policy_params = model.hyper_net(hn_in)
        logits = apply_generated_policy(
            o, policy_params, model.obs_size, model.policy_hidden, model.n_actions)
        critic_in = jnp.concatenate([new_h, belief, o])
        value = model.critic_head(critic_in).squeeze(-1)
        return logits, value

    logits, values = jax.vmap(forward_one)(aug, obs, gru_h)
    return ppo_loss_from_logits(
        logits, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)


def v28_elbo_loss_fn(model, elbo_data, rng_key, beta=1.0):
    """Supervised CE on task id + reconstruction + uniform-KL regularizer.

    elbo_data: 7-tuple (obs, actions, rewards, next_obs, prev_act_oh,
                        prev_rew, task_ids)
    """
    obs, actions, rewards, next_obs, prev_act_oh, prev_rew, task_ids = elbo_data

    def forward_trial(obs_s, act_s, rew_s, nobs_s, pa_s, pr_s, tid_s):
        def scan_fn(h, inp):
            o, pa, pr = inp
            aug = jnp.concatenate([o, pa, pr])
            h = model.gru_cell(aug, h)
            logits_post = model.posterior_net(h)
            return h, logits_post

        h0 = jnp.zeros(model.hidden_size)
        _, logits_post_seq = jax.lax.scan(scan_fn, h0, (obs_s, pa_s, pr_s))

        log_belief = jax.nn.log_softmax(logits_post_seq, axis=-1)
        belief = jnp.exp(log_belief)

        # Supervised CE on true task id
        targets = jax.nn.one_hot(tid_s, model.n_tasks)
        class_loss = -jnp.mean(jnp.sum(targets * log_belief, axis=-1))

        # Reconstruction: decoder sees belief as input
        act_oh = jax.nn.one_hot(act_s, model.n_actions)
        dec_in = jnp.concatenate([belief, obs_s, act_oh], axis=-1)
        dec_out = jax.vmap(model.decoder_net)(dec_in)
        r_hat = dec_out[:, 0]
        o_hat = dec_out[:, 1:]
        recon = jnp.mean((r_hat - rew_s) ** 2) + jnp.mean((o_hat - nobs_s) ** 2)

        # KL(belief || uniform) = log K - H(belief)
        log_k = jnp.log(jnp.float32(model.n_tasks))
        entropy = -jnp.sum(belief * log_belief, axis=-1)
        kl_unif = jnp.mean(log_k - entropy)

        return (CLASSIFY_WEIGHT * class_loss +
                RECON_WEIGHT * recon +
                KL_UNIF_WEIGHT * kl_unif)

    trial_losses = jax.vmap(forward_trial,
                            in_axes=(1, 1, 1, 1, 1, 1, 1))(
        obs, actions, rewards, next_obs, prev_act_oh, prev_rew, task_ids)
    return beta * jnp.mean(trial_losses)
