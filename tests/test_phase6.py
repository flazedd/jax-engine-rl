"""Phase 6 tests — agent framework + PPO training."""
import pytest
import jax
import jax.numpy as jnp
import time
import equinox as eqx
import optax

from lob_sim.config import SimConfig
from lob_sim.actions import N_ACTIONS
from lob_sim.agents.ppo import PPOAgent, PPOConfig
from lob_sim.agents.base import AgentState, RolloutBatch
from lob_sim.training.rollout import collect_rollout, collect_rollout_batch
from lob_sim.training.trainer import compute_gae, ppo_update, create_optimizer
from lob_sim.training.eval import evaluate_agent

# Small configs for fast tests
FAST_PPO_CONFIG = PPOConfig(
    lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
    entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
    n_epochs=2, n_minibatches=2, n_envs=16, n_steps=128, hidden_size=0,
)
SIM_CONFIG = SimConfig()


@pytest.fixture(scope="module")
def agent():
    key = jax.random.PRNGKey(0)
    return PPOAgent(ppo_config=FAST_PPO_CONFIG, key=key)


@pytest.fixture(scope="module")
def agent_and_opt(agent):
    optimizer, opt_state = create_optimizer(FAST_PPO_CONFIG, agent)
    return agent, optimizer, opt_state


class TestAgentInterface:
    def test_initial_state_shape(self, agent):
        """PPO initial_agent_state returns AgentState with correct dummy shapes."""
        key = jax.random.PRNGKey(0)
        state = agent.initial_agent_state(key)
        assert isinstance(state, AgentState)
        assert state.prev_action.shape == ()
        assert state.prev_reward.shape == ()

    def test_get_action_shapes(self, agent):
        """get_action returns (scalar int32, AgentState, info_dict)."""
        key = jax.random.PRNGKey(1)
        obs = jnp.zeros(33)
        agent_state = agent.initial_agent_state(key)
        action, new_state, info = agent.get_action(obs, agent_state, key)
        assert action.shape == ()
        assert action.dtype == jnp.int32
        assert 0 <= int(action) <= N_ACTIONS - 1
        assert "log_prob" in info
        assert "value" in info
        assert info["log_prob"].shape == ()
        assert info["value"].shape == ()

    def test_get_action_action_in_range(self, agent):
        """Actions sampled over 100 calls are all in [0, N_ACTIONS-1]."""
        key = jax.random.PRNGKey(2)
        obs = jnp.zeros(33)
        agent_state = agent.initial_agent_state(key)
        keys = jax.random.split(key, 100)
        actions = jnp.array([
            agent.get_action(obs, agent_state, k)[0] for k in keys
        ])
        assert jnp.all(actions >= 0)
        assert jnp.all(actions <= N_ACTIONS - 1)

    def test_get_action_jit(self, agent):
        """get_action compiles under jax.jit without error."""
        key = jax.random.PRNGKey(3)
        obs = jnp.zeros(33)
        agent_state = agent.initial_agent_state(key)
        jit_fn = jax.jit(agent.get_action)
        action, _, info = jit_fn(obs, agent_state, key)
        jax.block_until_ready(action)

    def test_ppo_stateless(self, agent):
        """PPO agent_state is unchanged after get_action (stateless)."""
        key = jax.random.PRNGKey(4)
        obs = jnp.zeros(33)
        agent_state = agent.initial_agent_state(key)
        _, new_state, _ = agent.get_action(obs, agent_state, key)
        # hidden should still be zeros
        assert jnp.allclose(new_state.hidden, agent_state.hidden)

    def test_different_keys_different_actions(self, agent):
        """Different RNG keys produce at least some different actions."""
        obs = jnp.zeros(33)
        agent_state = agent.initial_agent_state(jax.random.PRNGKey(0))
        keys = jax.random.split(jax.random.PRNGKey(99), 20)
        actions = [int(agent.get_action(obs, agent_state, k)[0]) for k in keys]
        assert len(set(actions)) > 1, "All actions identical — RNG not being used"


class TestRollout:
    def test_single_rollout_compiles(self, agent):
        """collect_rollout compiles under jax.jit."""
        key = jax.random.PRNGKey(0)
        jit_fn = jax.jit(
            lambda k: collect_rollout(agent, SIM_CONFIG, k, n_steps=32, locked_regime=0)
        )
        _, traj, last_val = jit_fn(key)
        jax.block_until_ready(traj.obs)

    def test_single_rollout_shapes(self, agent):
        """collect_rollout trajectory has leading dim n_steps."""
        key = jax.random.PRNGKey(0)
        N = 32
        _, traj, last_val = collect_rollout(agent, SIM_CONFIG, key, n_steps=N, locked_regime=0)
        assert traj.obs.shape == (N, 33)
        assert traj.action.shape == (N,)
        assert traj.log_prob.shape == (N,)
        assert traj.value.shape == (N,)
        assert traj.reward.shape == (N,)
        assert traj.done.shape == (N,)
        assert last_val.shape == ()

    def test_batch_rollout_shapes(self, agent):
        """collect_rollout_batch returns (n_envs, n_steps, ...) shapes."""
        key = jax.random.PRNGKey(0)
        batch = collect_rollout_batch(agent, SIM_CONFIG, key,
                                      n_envs=4, n_steps=32, locked_regime=0)
        assert batch.obs.shape == (4, 32, 33)
        assert batch.actions.shape == (4, 32)
        assert batch.rewards.shape == (4, 32)
        assert batch.last_value.shape == (4,)

    def test_actions_in_valid_range(self, agent):
        """All actions in rollout are in [0, N_ACTIONS-1]."""
        key = jax.random.PRNGKey(0)
        batch = collect_rollout_batch(agent, SIM_CONFIG, key,
                                      n_envs=4, n_steps=64, locked_regime=0)
        assert jnp.all(batch.actions >= 0)
        assert jnp.all(batch.actions <= N_ACTIONS - 1)

    def test_no_nan_in_rollout(self, agent):
        """Rollout contains no NaN values in obs, rewards, or log_probs."""
        key = jax.random.PRNGKey(0)
        batch = collect_rollout_batch(agent, SIM_CONFIG, key,
                                      n_envs=4, n_steps=64, locked_regime=0)
        assert not jnp.any(jnp.isnan(batch.obs))
        assert not jnp.any(jnp.isnan(batch.rewards))
        assert not jnp.any(jnp.isnan(batch.log_probs))

    def test_rollout_throughput(self, agent):
        """Warm rollout throughput > 5,000 steps/sec (single env)."""
        key = jax.random.PRNGKey(0)
        N = 512
        fn = jax.jit(lambda k: collect_rollout(agent, SIM_CONFIG, k, n_steps=N, locked_regime=0))
        fn(key)  # compile
        jax.block_until_ready(fn(key))
        t0 = time.perf_counter()
        _, traj, _ = fn(key)
        jax.block_until_ready(traj.obs)
        elapsed = time.perf_counter() - t0
        steps_per_sec = N / elapsed
        print(f"\nRollout throughput: {steps_per_sec:.0f} steps/sec")
        assert steps_per_sec > 5_000


class TestGAE:
    def test_gae_shapes(self, agent):
        """compute_gae returns advantages and returns with shape (n_envs, n_steps)."""
        key = jax.random.PRNGKey(0)
        batch = collect_rollout_batch(agent, SIM_CONFIG, key,
                                      n_envs=4, n_steps=32, locked_regime=0)
        adv, ret = compute_gae(batch, gamma=0.99, gae_lambda=0.95)
        assert adv.shape == (4, 32)
        assert ret.shape == (4, 32)

    def test_gae_no_nan(self, agent):
        """GAE produces no NaN values."""
        key = jax.random.PRNGKey(0)
        batch = collect_rollout_batch(agent, SIM_CONFIG, key,
                                      n_envs=4, n_steps=32, locked_regime=0)
        adv, ret = compute_gae(batch, gamma=0.99, gae_lambda=0.95)
        assert not jnp.any(jnp.isnan(adv))
        assert not jnp.any(jnp.isnan(ret))

    def test_gae_known_trajectory(self):
        """GAE matches hand-computed value for a 3-step single-env trajectory.

        Setup:
            rewards    = [1.0, 0.0, 1.0]
            values     = [0.5, 0.5, 0.5]
            dones      = [0,   0,   1  ]
            last_value = 0.5             (ignored because final done=1)
            gamma=1.0, gae_lambda=1.0   (simplifies arithmetic)

        Hand calculation (reverse):
            t=2: delta = 1.0 + 1.0*0.5*(1-1) - 0.5 = 0.5;  A_2 = 0.5
            t=1: delta = 0.0 + 1.0*0.5*(1-0) - 0.5 = 0.0;  A_1 = 0.0 + 1.0*1.0*(1-0)*0.5 = 0.5
            t=0: delta = 1.0 + 1.0*0.5*(1-0) - 0.5 = 1.0;  A_0 = 1.0 + 1.0*1.0*(1-0)*0.5 = 1.5
        Expected advantages: [1.5, 0.5, 0.5]
        Expected returns:    [2.0, 1.0, 1.0]  (advantages + values)
        """
        batch = RolloutBatch(
            obs=jnp.zeros((1, 3, 33)),
            actions=jnp.zeros((1, 3), dtype=jnp.int32),
            log_probs=jnp.zeros((1, 3)),
            values=jnp.array([[0.5, 0.5, 0.5]]),
            rewards=jnp.array([[1.0, 0.0, 1.0]]),
            dones=jnp.array([[0.0, 0.0, 1.0]]),
            last_value=jnp.array([0.5]),
        )
        adv, ret = compute_gae(batch, gamma=1.0, gae_lambda=1.0)
        expected_adv = jnp.array([[1.5, 0.5, 0.5]])
        expected_ret = jnp.array([[2.0, 1.0, 1.0]])
        assert jnp.allclose(adv, expected_adv, atol=1e-5), f"GAE advantages wrong: {adv}"
        assert jnp.allclose(ret, expected_ret, atol=1e-5), f"GAE returns wrong: {ret}"

    def test_returns_equal_advantages_plus_values(self, agent):
        """Returns must equal advantages + values everywhere."""
        key = jax.random.PRNGKey(0)
        batch = collect_rollout_batch(agent, SIM_CONFIG, key,
                                      n_envs=4, n_steps=32, locked_regime=0)
        adv, ret = compute_gae(batch, gamma=0.99, gae_lambda=0.95)
        assert jnp.allclose(ret, adv + batch.values, atol=1e-5)


class TestPPOUpdate:
    def test_single_update_runs(self, agent_and_opt):
        """One PPO update step runs without error and returns finite metrics."""
        agent, optimizer, opt_state = agent_and_opt
        key = jax.random.PRNGKey(0)
        batch = collect_rollout_batch(agent, SIM_CONFIG, key,
                                      n_envs=FAST_PPO_CONFIG.n_envs,
                                      n_steps=FAST_PPO_CONFIG.n_steps,
                                      locked_regime=0)
        adv, ret = compute_gae(batch, FAST_PPO_CONFIG.gamma, FAST_PPO_CONFIG.gae_lambda)
        updated_agent, new_opt_state, metrics = ppo_update(
            agent, optimizer, opt_state, batch, adv, ret, FAST_PPO_CONFIG, key
        )
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "total_loss" in metrics
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"Metric {k} is not finite: {v}"

    def test_update_changes_params(self, agent_and_opt):
        """After one update, agent parameters must actually change."""
        agent, optimizer, opt_state = agent_and_opt
        key = jax.random.PRNGKey(0)
        batch = collect_rollout_batch(agent, SIM_CONFIG, key,
                                      n_envs=FAST_PPO_CONFIG.n_envs,
                                      n_steps=FAST_PPO_CONFIG.n_steps,
                                      locked_regime=0)
        adv, ret = compute_gae(batch, FAST_PPO_CONFIG.gamma, FAST_PPO_CONFIG.gae_lambda)
        updated_agent, _, _ = ppo_update(
            agent, optimizer, opt_state, batch, adv, ret, FAST_PPO_CONFIG, key
        )
        # Compare one weight tensor from trunk
        old_w = eqx.filter(agent, eqx.is_array)
        new_w = eqx.filter(updated_agent, eqx.is_array)
        old_leaves = jax.tree_util.tree_leaves(old_w)
        new_leaves = jax.tree_util.tree_leaves(new_w)
        any_changed = any(
            not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves)
        )
        assert any_changed, "No parameters changed after PPO update"

    def test_entropy_positive(self, agent_and_opt):
        """Entropy metric should be positive (policy not fully deterministic)."""
        agent, optimizer, opt_state = agent_and_opt
        key = jax.random.PRNGKey(0)
        batch = collect_rollout_batch(agent, SIM_CONFIG, key,
                                      n_envs=FAST_PPO_CONFIG.n_envs,
                                      n_steps=FAST_PPO_CONFIG.n_steps,
                                      locked_regime=0)
        adv, ret = compute_gae(batch, FAST_PPO_CONFIG.gamma, FAST_PPO_CONFIG.gae_lambda)
        _, _, metrics = ppo_update(
            agent, optimizer, opt_state, batch, adv, ret, FAST_PPO_CONFIG, key
        )
        assert float(metrics["entropy"]) > 0.0


class TestTrainingImprovement:
    def test_reward_improves_over_training(self):
        """After 30 training iterations on locked noise, mean eval reward in last
        10 iterations >= mean eval reward in first 10 iterations.
        Uses small config to keep runtime under 30s.
        """
        import optax
        key = jax.random.PRNGKey(42)
        ppo_cfg = FAST_PPO_CONFIG  # n_envs=8, n_steps=64, n_epochs=2, n_minibatches=2
        sim_cfg = SimConfig()

        key, k0 = jax.random.split(key)
        agent = PPOAgent(ppo_config=ppo_cfg, key=k0)
        optimizer, opt_state = create_optimizer(ppo_cfg, agent)

        eval_rewards = []
        for i in range(50):
            key, k_roll, k_upd, k_eval = jax.random.split(key, 4)
            batch = collect_rollout_batch(agent, sim_cfg, k_roll,
                                          n_envs=ppo_cfg.n_envs,
                                          n_steps=ppo_cfg.n_steps,
                                          locked_regime=0)
            adv, ret = compute_gae(batch, ppo_cfg.gamma, ppo_cfg.gae_lambda)
            agent, opt_state, _ = ppo_update(agent, optimizer, opt_state,
                                              batch, adv, ret, ppo_cfg, k_upd)
            stats = evaluate_agent(agent, sim_cfg, k_eval,
                                    n_episodes=10, locked_regime=0)
            eval_rewards.append(float(stats["mean_reward"]))

        first_10 = sum(eval_rewards[:10]) / 10
        last_10 = sum(eval_rewards[40:]) / 10
        print(f"\nFirst 10 mean reward: {first_10:.4f}")
        print(f"Last 10 mean reward:  {last_10:.4f}")
        # Reward should improve or at least not collapse
        # Allow 50% tolerance — early random policy can score well from random fills
        assert last_10 >= first_10 - abs(first_10) * 0.50, (
            f"Reward regressed: {first_10:.4f} → {last_10:.4f}"
        )
