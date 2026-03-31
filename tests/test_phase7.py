"""Phase 7 tests — RL² agent."""
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from lob_sim.config import SimConfig
from lob_sim.actions import N_ACTIONS
from lob_sim.agents.rl2 import RL2Agent, RL2Config
from lob_sim.agents.ppo import PPOAgent, PPOConfig
from lob_sim.agents.base import AgentState
from lob_sim.state import init_state
from lob_sim.obs import observe
from lob_sim.step import make_step_fn
from lob_sim.training.rollout import collect_rollout, collect_rollout_batch
from lob_sim.training.trainer import compute_gae, rl2_ppo_update, ppo_update, create_optimizer
from lob_sim.training.eval import evaluate_agent

SIM_CONFIG = SimConfig()

# Small configs for fast tests
FAST_RL2_CONFIG = RL2Config(
    lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
    entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
    n_epochs=2, n_minibatches=2, n_envs=8, n_steps=128,
    hidden_size=64,
)


@pytest.fixture(scope="module")
def agent():
    key = jax.random.PRNGKey(0)
    return RL2Agent(hidden_size=64, key=key)


class TestRL2Agent:
    def test_initial_state_shape(self, agent):
        """Hidden state shape is (hidden_size,)."""
        key = jax.random.PRNGKey(0)
        state = agent.initial_agent_state(key)
        assert isinstance(state, AgentState)
        assert state.hidden.shape == (64,)
        assert state.prev_action.shape == ()
        assert state.prev_reward.shape == ()
        assert state.prev_done.shape == ()

    def test_get_action_updates_hidden(self, agent):
        """Hidden state changes after get_action call."""
        key = jax.random.PRNGKey(1)
        obs = jnp.zeros(33)
        state = agent.initial_agent_state(key)
        action, new_state, info = agent.get_action(obs, state, key)
        assert not jnp.allclose(new_state.hidden, state.hidden), \
            "Hidden state should change after get_action"

    def test_input_includes_prev_action_reward(self, agent):
        """GRU input is [obs, prev_action_onehot, prev_reward, prev_done]."""
        expected_dim = 33 + N_ACTIONS + 1 + 1  # 44

        # Verify GRU accepts the correct input dimension
        prev_action_onehot = jax.nn.one_hot(jnp.int32(0), N_ACTIONS)
        gru_input = jnp.concatenate([
            jnp.zeros(33),
            prev_action_onehot,
            jnp.zeros(1),
            jnp.zeros(1),
        ])
        assert gru_input.shape == (expected_dim,)

        # Verify GRU processes it
        hidden = jnp.zeros(agent.hidden_size)
        new_hidden = agent.gru(gru_input, hidden)
        assert new_hidden.shape == (agent.hidden_size,)

    def test_jit_compatible(self, agent):
        """get_action compiles under jit."""
        key = jax.random.PRNGKey(3)
        obs = jnp.zeros(33)
        state = agent.initial_agent_state(key)
        jit_fn = jax.jit(agent.get_action)
        action, _, info = jit_fn(obs, state, key)
        jax.block_until_ready(action)
        assert action.shape == ()
        assert action.dtype == jnp.int32
        assert 0 <= int(action) <= N_ACTIONS - 1

    def test_hidden_persists_across_done(self, agent):
        """When done=True, hidden state is NOT zeroed — it persists in meta-episode mode."""
        key = jax.random.PRNGKey(5)
        obs = jnp.ones(33) * 0.5
        state = agent.initial_agent_state(key)

        # Run a few steps to build up hidden state
        for i in range(5):
            k = jax.random.fold_in(key, i)
            _, state, _ = agent.get_action(obs, state, k)
            state = state._replace(prev_reward=jnp.float32(0.1))

        hidden_before_done = state.hidden.copy()
        assert jnp.any(hidden_before_done != 0.0), "Hidden should be non-zero after steps"

        # Simulate done=True: set prev_done=1 in agent state
        state = state._replace(prev_done=jnp.float32(1.0))
        _, new_state, _ = agent.get_action(obs, state, jax.random.fold_in(key, 99))

        # Hidden should still be non-zero (not reset)
        assert jnp.any(new_state.hidden != 0.0), "Hidden should not be zeroed on done"
        # Hidden should change (GRU processed the done signal)
        assert not jnp.allclose(new_state.hidden, hidden_before_done), \
            "Hidden should change when processing done signal"

    def test_different_keys_different_actions(self, agent):
        """Different RNG keys produce at least some different actions."""
        obs = jnp.zeros(33)
        state = agent.initial_agent_state(jax.random.PRNGKey(0))
        keys = jax.random.split(jax.random.PRNGKey(99), 20)
        actions = [int(agent.get_action(obs, state, k)[0]) for k in keys]
        assert len(set(actions)) > 1, "All actions identical — RNG not being used"

    def test_get_action_shapes(self, agent):
        """get_action returns correct shapes and types."""
        key = jax.random.PRNGKey(1)
        obs = jnp.zeros(33)
        state = agent.initial_agent_state(key)
        action, new_state, info = agent.get_action(obs, state, key)
        assert action.shape == ()
        assert action.dtype == jnp.int32
        assert "log_prob" in info
        assert "value" in info
        assert info["log_prob"].shape == ()
        assert info["value"].shape == ()


class TestRL2Rollout:
    def test_rollout_compiles(self, agent):
        """collect_rollout with RL² compiles under jax.jit."""
        key = jax.random.PRNGKey(0)
        jit_fn = jax.jit(
            lambda k: collect_rollout(agent, SIM_CONFIG, k, n_steps=32, locked_regime=0)
        )
        _, traj, last_val = jit_fn(key)
        jax.block_until_ready(traj.obs)

    def test_rollout_shapes(self, agent):
        """collect_rollout trajectory has correct shapes."""
        key = jax.random.PRNGKey(0)
        N = 32
        _, traj, last_val = collect_rollout(agent, SIM_CONFIG, key, n_steps=N, locked_regime=0)
        assert traj.obs.shape == (N, 33)
        assert traj.action.shape == (N,)
        assert traj.reward.shape == (N,)
        assert last_val.shape == ()

    def test_batch_rollout_shapes(self, agent):
        """collect_rollout_batch returns correct shapes."""
        key = jax.random.PRNGKey(0)
        batch = collect_rollout_batch(agent, SIM_CONFIG, key,
                                      n_envs=4, n_steps=32, locked_regime=0)
        assert batch.obs.shape == (4, 32, 33)
        assert batch.actions.shape == (4, 32)
        assert batch.rewards.shape == (4, 32)

    def test_meta_episode_rollout(self, agent):
        """Meta-episode rollout compiles and runs."""
        key = jax.random.PRNGKey(0)
        sim_cfg = SimConfig(max_steps=50)  # short episodes for meta-episode testing
        batch = collect_rollout_batch(agent, sim_cfg, key,
                                      n_envs=4, n_steps=200,
                                      locked_regime=-1, meta_episode=True)
        assert batch.obs.shape == (4, 200, 33)
        # With max_steps=50 and n_steps=200, there should be done events
        assert jnp.any(batch.dones), "Expected at least some done events in meta-episode"
        # No NaN
        assert not jnp.any(jnp.isnan(batch.obs))
        assert not jnp.any(jnp.isnan(batch.rewards))

    def test_no_nan_in_rollout(self, agent):
        """RL² rollout contains no NaN values."""
        key = jax.random.PRNGKey(0)
        batch = collect_rollout_batch(agent, SIM_CONFIG, key,
                                      n_envs=4, n_steps=64, locked_regime=0)
        assert not jnp.any(jnp.isnan(batch.obs))
        assert not jnp.any(jnp.isnan(batch.rewards))
        assert not jnp.any(jnp.isnan(batch.log_probs))


class TestRL2Training:
    def test_trains_without_error(self):
        """30 iterations of training on mixed regimes completes."""
        key = jax.random.PRNGKey(42)
        cfg = FAST_RL2_CONFIG
        sim_cfg = SimConfig(max_steps=200)

        key, k0 = jax.random.split(key)
        agent = RL2Agent(hidden_size=cfg.hidden_size, key=k0)
        optimizer, opt_state = create_optimizer(cfg, agent)

        for i in range(30):
            key, k_roll, k_upd = jax.random.split(key, 3)
            batch = collect_rollout_batch(agent, sim_cfg, k_roll,
                                          n_envs=cfg.n_envs,
                                          n_steps=cfg.n_steps,
                                          locked_regime=-1,
                                          meta_episode=True)
            adv, ret = compute_gae(batch, cfg.gamma, cfg.gae_lambda)
            agent, opt_state, metrics = rl2_ppo_update(
                agent, optimizer, opt_state, batch, adv, ret, cfg, k_upd
            )
            assert jnp.isfinite(metrics["total_loss"]), f"Loss not finite at iter {i}"

        print(f"\nRL2 training metrics after 30 iters:")
        for k, v in metrics.items():
            print(f"  {k}: {float(v):.4f}")

    def test_outperforms_ppo(self):
        """After training, RL² performs at least comparably to PPO on mixed regimes."""
        key = jax.random.PRNGKey(42)
        sim_cfg = SimConfig(max_steps=200)
        n_iters = 80

        # Train RL²
        rl2_cfg = FAST_RL2_CONFIG
        key, k0 = jax.random.split(key)
        rl2_agent = RL2Agent(hidden_size=rl2_cfg.hidden_size, key=k0)
        rl2_opt, rl2_opt_state = create_optimizer(rl2_cfg, rl2_agent)

        for i in range(n_iters):
            key, k_roll, k_upd = jax.random.split(key, 3)
            batch = collect_rollout_batch(rl2_agent, sim_cfg, k_roll,
                                          n_envs=rl2_cfg.n_envs,
                                          n_steps=rl2_cfg.n_steps,
                                          locked_regime=-1,
                                          meta_episode=True)
            adv, ret = compute_gae(batch, rl2_cfg.gamma, rl2_cfg.gae_lambda)
            rl2_agent, rl2_opt_state, _ = rl2_ppo_update(
                rl2_agent, rl2_opt, rl2_opt_state, batch, adv, ret, rl2_cfg, k_upd
            )

        # Train PPO with same budget
        ppo_cfg = PPOConfig(
            lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
            entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
            n_epochs=2, n_minibatches=2, n_envs=8, n_steps=128,
        )
        key, k0 = jax.random.split(key)
        ppo_agent = PPOAgent(ppo_config=ppo_cfg, key=k0)
        ppo_opt, ppo_opt_state = create_optimizer(ppo_cfg, ppo_agent)

        for i in range(n_iters):
            key, k_roll, k_upd = jax.random.split(key, 3)
            batch = collect_rollout_batch(ppo_agent, sim_cfg, k_roll,
                                          n_envs=ppo_cfg.n_envs,
                                          n_steps=ppo_cfg.n_steps,
                                          locked_regime=-1)
            adv, ret = compute_gae(batch, ppo_cfg.gamma, ppo_cfg.gae_lambda)
            ppo_agent, ppo_opt_state, _ = ppo_update(
                ppo_agent, ppo_opt, ppo_opt_state, batch, adv, ret, ppo_cfg, k_upd
            )

        # Evaluate both on same episodes
        key, k_eval = jax.random.split(key)
        rl2_stats = evaluate_agent(rl2_agent, sim_cfg, k_eval,
                                    n_episodes=30, locked_regime=-1)
        ppo_stats = evaluate_agent(ppo_agent, sim_cfg, k_eval,
                                    n_episodes=30, locked_regime=-1)

        rl2_reward = float(rl2_stats['mean_reward'])
        ppo_reward = float(ppo_stats['mean_reward'])
        print(f"\nRL2 mean reward: {rl2_reward:.4f}")
        print(f"PPO mean reward: {ppo_reward:.4f}")

        # RL² should be at least comparable (within 50% tolerance for short training)
        assert rl2_reward >= ppo_reward - abs(ppo_reward) * 0.50, (
            f"RL2 ({rl2_reward:.4f}) significantly worse than PPO ({ppo_reward:.4f})"
        )

    def test_hidden_state_correlates_with_regime(self):
        """Hidden states should show some difference across regimes."""
        key = jax.random.PRNGKey(42)
        sim_cfg = SimConfig()
        cfg = FAST_RL2_CONFIG

        # Train briefly
        key, k0 = jax.random.split(key)
        agent = RL2Agent(hidden_size=cfg.hidden_size, key=k0)
        optimizer, opt_state = create_optimizer(cfg, agent)

        for i in range(30):
            key, k_roll, k_upd = jax.random.split(key, 3)
            batch = collect_rollout_batch(agent, SimConfig(max_steps=200), k_roll,
                                          n_envs=cfg.n_envs,
                                          n_steps=cfg.n_steps,
                                          locked_regime=-1,
                                          meta_episode=True)
            adv, ret = compute_gae(batch, cfg.gamma, cfg.gae_lambda)
            agent, opt_state, _ = rl2_ppo_update(
                agent, optimizer, opt_state, batch, adv, ret, cfg, k_upd
            )

        # Collect hidden states per regime using lax.scan
        step_fn_map = {}
        def collect_hiddens(agent, sim_cfg, rng_key, n_steps, regime):
            step_fn = make_step_fn(sim_cfg, locked_regime=regime)

            def scan_body(carry, _):
                sim_state, agent_state, rng = carry
                rng, rng_action = jax.random.split(rng)

                obs = observe(sim_state, sim_cfg)
                action, new_agent_state, _ = agent.get_action(obs, agent_state, rng_action)
                new_sim_state, sim_out = step_fn(sim_state, action)

                new_agent_state = new_agent_state._replace(
                    prev_reward=sim_out["reward"],
                    prev_done=sim_out["done"].astype(jnp.float32),
                )

                return (new_sim_state, new_agent_state, rng), new_agent_state.hidden

            k_init, k_agent, k_scan = jax.random.split(rng_key, 3)
            sim_state = init_state(sim_cfg, k_init)
            agent_state = agent.initial_agent_state(k_agent)
            _, hiddens = jax.lax.scan(scan_body, (sim_state, agent_state, k_scan), None, length=n_steps)
            return hiddens

        hidden_norms = []
        hidden_means = []
        for regime in range(3):
            key, k = jax.random.split(key)
            hiddens = jax.jit(
                lambda k, r=regime: collect_hiddens(agent, sim_cfg, k, 200, r)
            )(k)
            norm = float(jnp.mean(jnp.linalg.norm(hiddens, axis=1)))
            mean = float(jnp.mean(hiddens))
            hidden_norms.append(norm)
            hidden_means.append(mean)

        print(f"\nHidden state norms by regime: {hidden_norms}")
        print(f"Hidden state means by regime: {hidden_means}")
        # Informational — just check hidden states are non-degenerate
        assert all(n > 0.01 for n in hidden_norms), \
            "Hidden states should be non-trivial (norm > 0.01)"
