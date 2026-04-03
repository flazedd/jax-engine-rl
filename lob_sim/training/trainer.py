"""GAE computation and PPO update."""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from lob_sim.agents.base import RolloutBatch
from lob_sim.actions import N_ACTIONS


def compute_gae(batch: RolloutBatch, gamma: float, gae_lambda: float):
    """Compute GAE advantages and value targets.

    Args:
        batch: RolloutBatch with shapes (n_envs, n_steps, ...)
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: (n_envs, n_steps) float32
        returns:    (n_envs, n_steps) float32
    """
    n_envs, n_steps = batch.rewards.shape

    rewards = batch.rewards

    # Build next_values: shift values by 1, use last_value for final step
    next_values = jnp.concatenate(
        [batch.values[:, 1:], batch.last_value[:, None]], axis=1
    )

    def gae_step(gae_next, t_data):
        reward, value, next_value, done = t_data
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae_next
        return gae, gae

    # Reverse: process from T-1 down to 0
    # We need to reverse the time dimension, scan, then reverse back
    rewards_rev = jnp.flip(rewards, axis=1)
    values_rev = jnp.flip(batch.values, axis=1)
    next_values_rev = jnp.flip(next_values, axis=1)
    dones_rev = jnp.flip(batch.dones, axis=1)

    def per_env_gae(rewards_r, values_r, next_values_r, dones_r):
        _, advantages_rev = jax.lax.scan(
            gae_step,
            jnp.float32(0.0),
            (rewards_r, values_r, next_values_r, dones_r),
        )
        return jnp.flip(advantages_rev)

    advantages = jax.vmap(per_env_gae)(rewards_rev, values_rev, next_values_rev, dones_rev)
    returns = advantages + batch.values
    return advantages, returns


def create_optimizer(ppo_config, agent):
    """Create optimizer and initial opt_state."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(ppo_config.max_grad_norm),
        optax.adam(ppo_config.lr),
    )
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_array))
    return optimizer, opt_state


def ppo_update(agent, optimizer, opt_state, batch, advantages, returns, ppo_config, rng_key):
    """Run n_epochs x n_minibatches of PPO updates.

    Returns: (updated_agent, updated_opt_state, metrics_dict)
    """
    n_envs, n_steps = batch.rewards.shape
    total_size = n_envs * n_steps
    mb_size = total_size // ppo_config.n_minibatches

    # Flatten batch
    flat_obs = batch.obs.reshape(total_size, -1)
    flat_actions = batch.actions.reshape(total_size)
    flat_old_log_probs = batch.log_probs.reshape(total_size)
    flat_values = batch.values.reshape(total_size)
    flat_advantages = advantages.reshape(total_size)
    flat_returns = returns.reshape(total_size)

    # Normalize advantages
    adv_mean = jnp.mean(flat_advantages)
    adv_std = jnp.std(flat_advantages) + 1e-8
    flat_advantages = (flat_advantages - adv_mean) / adv_std

    def loss_fn(agent, obs_mb, actions_mb, old_lp_mb, adv_mb, ret_mb):
        def forward_one(obs, action):
            trunk_out = agent.trunk(obs)
            logits = agent.policy_head(trunk_out)
            value = agent.value_head(trunk_out)[0]
            log_probs_all = jax.nn.log_softmax(logits)
            log_prob = log_probs_all[action]
            probs = jnp.exp(log_probs_all)
            entropy = -jnp.sum(probs * log_probs_all)
            return log_prob, value, entropy

        new_log_probs, new_values, entropies = jax.vmap(forward_one)(obs_mb, actions_mb)

        # Policy loss (clipped surrogate)
        ratio = jnp.exp(new_log_probs - old_lp_mb)
        surr1 = ratio * adv_mb
        surr2 = jnp.clip(ratio, 1.0 - ppo_config.clip_eps, 1.0 + ppo_config.clip_eps) * adv_mb
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # Value loss
        value_loss = jnp.mean((new_values - ret_mb) ** 2)

        # Entropy
        entropy = jnp.mean(entropies)

        total_loss = policy_loss + ppo_config.value_coef * value_loss - ppo_config.entropy_coef * entropy

        return total_loss, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy, "total_loss": total_loss}

    def epoch_step(carry, epoch_key):
        agent, opt_state = carry
        perm = jax.random.permutation(epoch_key, total_size)

        def minibatch_step(carry2, mb_idx):
            agent, opt_state = carry2
            start = mb_idx * mb_size
            idx = jax.lax.dynamic_slice(perm, (start,), (mb_size,))

            obs_mb = flat_obs[idx]
            actions_mb = flat_actions[idx]
            old_lp_mb = flat_old_log_probs[idx]
            adv_mb = flat_advantages[idx]
            ret_mb = flat_returns[idx]

            grads, metrics = jax.grad(loss_fn, has_aux=True)(
                agent, obs_mb, actions_mb, old_lp_mb, adv_mb, ret_mb
            )

            grads = eqx.filter(grads, eqx.is_array)
            updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(agent, eqx.is_array))
            new_agent = eqx.apply_updates(agent, updates)

            return (new_agent, new_opt_state), metrics

        mb_indices = jnp.arange(ppo_config.n_minibatches)
        (agent, opt_state), all_metrics = jax.lax.scan(
            minibatch_step, (agent, opt_state), mb_indices
        )

        # Average metrics across minibatches
        avg_metrics = jax.tree.map(lambda x: jnp.mean(x), all_metrics)
        return (agent, opt_state), avg_metrics

    epoch_keys = jax.random.split(rng_key, ppo_config.n_epochs)
    (agent, opt_state), all_epoch_metrics = jax.lax.scan(
        epoch_step, (agent, opt_state), epoch_keys
    )

    # Average metrics across epochs
    final_metrics = jax.tree.map(lambda x: jnp.mean(x), all_epoch_metrics)
    return agent, opt_state, final_metrics


def rl2_ppo_update(agent, optimizer, opt_state, batch, advantages, returns, config, rng_key):
    """PPO update for RL² agent — processes trajectories sequentially through GRU.

    Minibatches are over environments (not shuffled across time).
    """
    n_envs, n_steps = batch.rewards.shape
    mb_size = n_envs // config.n_minibatches

    # Normalize advantages
    flat_adv = advantages.reshape(-1)
    adv_mean = jnp.mean(flat_adv)
    adv_std = jnp.std(flat_adv) + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    def loss_fn(agent, obs, actions, old_log_probs, advs, rets, rewards, dones):
        def process_env(obs_seq, action_seq, old_lp_seq, adv_seq, ret_seq, reward_seq, done_seq):
            def gru_step(carry, t_data):
                hidden, prev_action, prev_reward, prev_done = carry
                obs_t, action_t, reward_t, done_t = t_data

                prev_action_onehot = jax.nn.one_hot(prev_action, N_ACTIONS)
                gru_input = jnp.concatenate([
                    obs_t,
                    prev_action_onehot,
                    jnp.atleast_1d(prev_reward),
                    jnp.atleast_1d(prev_done),
                ])
                new_hidden = agent.gru(gru_input, hidden)

                logits = agent.policy_head(new_hidden)
                value = agent.value_head(new_hidden)[0]
                log_probs_all = jax.nn.log_softmax(logits)
                log_prob = log_probs_all[action_t]
                probs = jnp.exp(log_probs_all)
                entropy = -jnp.sum(probs * log_probs_all)

                new_carry = (new_hidden, action_t, reward_t, done_t.astype(jnp.float32))
                return new_carry, (log_prob, value, entropy)

            init_carry = (
                jnp.zeros(agent.hidden_size),
                jnp.int32(0),
                jnp.float32(0.0),
                jnp.float32(0.0),
            )
            _, (new_log_probs, new_values, entropies) = jax.lax.scan(
                gru_step, init_carry,
                (obs_seq, action_seq, reward_seq, done_seq),
            )

            # PPO loss per environment
            ratio = jnp.exp(new_log_probs - old_lp_seq)
            surr1 = ratio * adv_seq
            surr2 = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv_seq
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
            value_loss = jnp.mean((new_values - ret_seq) ** 2)
            entropy_val = jnp.mean(entropies)

            return policy_loss, value_loss, entropy_val

        # vmap over environments in this minibatch
        policy_losses, value_losses, entropies = jax.vmap(process_env)(
            obs, actions, old_log_probs, advs, rets, rewards, dones
        )

        policy_loss = jnp.mean(policy_losses)
        value_loss = jnp.mean(value_losses)
        entropy = jnp.mean(entropies)
        total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

        return total_loss, {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
        }

    def epoch_step(carry, epoch_key):
        agent, opt_state = carry
        perm = jax.random.permutation(epoch_key, n_envs)

        def minibatch_step(carry2, mb_idx):
            agent, opt_state = carry2
            start = mb_idx * mb_size
            idx = jax.lax.dynamic_slice(perm, (start,), (mb_size,))

            grads, metrics = jax.grad(loss_fn, has_aux=True)(
                agent,
                batch.obs[idx], batch.actions[idx], batch.log_probs[idx],
                advantages[idx], returns[idx],
                batch.rewards[idx], batch.dones[idx],
            )

            grads = eqx.filter(grads, eqx.is_array)
            updates, new_opt_state = optimizer.update(
                grads, opt_state, eqx.filter(agent, eqx.is_array)
            )
            new_agent = eqx.apply_updates(agent, updates)

            return (new_agent, new_opt_state), metrics

        mb_indices = jnp.arange(config.n_minibatches)
        (agent, opt_state), all_metrics = jax.lax.scan(
            minibatch_step, (agent, opt_state), mb_indices
        )

        avg_metrics = jax.tree.map(lambda x: jnp.mean(x), all_metrics)
        return (agent, opt_state), avg_metrics

    epoch_keys = jax.random.split(rng_key, config.n_epochs)
    (agent, opt_state), all_epoch_metrics = jax.lax.scan(
        epoch_step, (agent, opt_state), epoch_keys
    )

    final_metrics = jax.tree.map(lambda x: jnp.mean(x), all_epoch_metrics)
    return agent, opt_state, final_metrics
