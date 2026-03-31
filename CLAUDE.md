# JAX LOB Simulator + RL Agents — Phased Build Spec

## Commands

- Always use `uv run` to execute Python: `uv run pytest tests/test_phaseN.py -v`
- Run plot scripts: `uv run python scripts/plot_regimes.py`
- Install deps: `uv sync`
- Never call python or pytest directly — always prefix with `uv run`
```

Then in your prompt:
```
Implement Phase 3 from CLAUDE.md. Run uv run pytest tests/test_phase3.py -v and fix failures. Then verify previous phases still pass with uv run pytest tests/test_phase1.py tests/test_phase2.py -v.

## Overview

Build a JAX-native limit order book simulator with hidden Markov regimes, then train RL agents (PPO, RL², VariBAD, VariBAD+HyperNetwork) that learn to adapt to regime switches.

**Build order**: complete each phase and verify its success criteria before starting the next. Each phase is designed to be a single Claude Code session.

## Global Constraints

- **Python 3.12**, managed via `uv`
- **JAX only** — no PyTorch, no TensorFlow
- All simulator functions must be **pure functions** over JAX arrays
- **No Python control flow** inside JIT-traced code — use `jnp.where`, `lax.cond`, `lax.scan`
- All array shapes must be **statically known** at compile time
- Target: CPU-first on macOS (Apple Silicon), but must work on GPU with zero code changes
- Use `typing.NamedTuple` (not dataclass) for state and config — automatically pytree-compatible

## Global Pitfalls

These apply to every phase. Keep them in mind throughout:

1. **RNG key reuse**: every `jax.random.*` call needs its own key. Split upfront into as many subkeys as needed. Reusing keys gives correlated samples.
2. **`jnp.roll` wrap-around**: after rolling, values that wrap from the other end are stale. Always mask to zero: `jnp.where(jnp.arange(n) >= n - shift, 0.0, rolled)`.
3. **No Python `if`/`for` inside JIT-traced functions**: use `jnp.where`, `lax.scan`, `lax.fori_loop`.
4. **Division by zero guards**: use `jnp.where(denom > 0, num / denom, 0.0)`.

---

## Dependencies

Phase 1–5 (simulator):
```toml
[project]
requires-python = "==3.12.*"
dependencies = ["jax", "jaxlib", "matplotlib", "pytest"]
```

Phase 6+ adds (when you get there):
```toml
dependencies = ["jax", "jaxlib", "matplotlib", "pytest", "optax", "distrax", "equinox"]
```

---

## Final Architecture (for reference — build incrementally)

```
lob_sim/
├── __init__.py
├── config.py           # SimConfig NamedTuple
├── regime.py           # HMM transition, per-regime params
├── state.py            # OrderBookState + init_state()
├── matching.py         # fill_market_buy(), fill_market_sell()
├── background.py       # generate_background_flow()
├── obs.py              # observe() → flat array
├── reward.py           # compute_reward()
├── actions.py          # Discrete action table (25 actions)
├── step.py             # make_step_fn(), run_episode()
├── agents/
│   ├── __init__.py
│   ├── base.py         # Agent protocol
│   ├── networks.py     # MLP, GRU, HyperNetwork
│   ├── ppo.py          # Vanilla PPO baseline
│   ├── rl2.py          # RL² (recurrent baseline)
│   ├── varibad.py      # VariBAD (VAE + belief-conditioned policy)
│   └── varibad_hyper.py # VariBAD + HyperNetwork
├── training/
│   ├── __init__.py
│   ├── rollout.py      # collect_rollout() via lax.scan
│   ├── trainer.py      # Training loop
│   ├── vae_buffer.py   # Trajectory replay for VAE
│   ├── eval.py         # Evaluation harness
│   └── logger.py       # Metrics logging
tests/
├── test_phase1.py
├── test_phase2.py
├── test_phase3.py
├── test_phase4.py
├── test_phase5.py
├── test_phase6.py
├── test_phase7.py
├── test_phase8.py
├── test_phase9.py
└── test_phase10.py
scripts/
├── plot_sanity.py
├── plot_regimes.py
├── plot_divergence.py
├── train.py
├── eval_agent.py
└── compare_agents.py
```

---
---

# PHASE 1 — Core Book Mechanics

## Scope

Build the orderbook state representation, matching engine, and basic background order flow. No regimes, no agent logic, no step loop yet — just the building blocks.

## Files to Create

- `lob_sim/__init__.py` (empty for now)
- `lob_sim/config.py`
- `lob_sim/state.py`
- `lob_sim/matching.py`
- `lob_sim/background.py`
- `tests/test_phase1.py`

## Specifications

### `config.py` — SimConfig

A `typing.NamedTuple` with all hyperparameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_levels` | int | 100 | Price levels each side of mid |
| `tick_size` | float | 0.01 | Price increment per level |
| `limit_order_rate` | float | 0.5 | Base arrival rate per level per step |
| `limit_order_size` | float | 1.0 | Fixed size of background limit orders |
| `market_buy_prob` | float | 0.15 | Probability of market buy per step |
| `market_sell_prob` | float | 0.15 | Probability of market sell per step |
| `market_order_size_min` | float | 1.0 | Market order size lower bound |
| `market_order_size_max` | float | 5.0 | Market order size upper bound |
| `cancel_prob` | float | 0.02 | Per-level cancellation probability per step |
| `depth_decay` | float | 0.05 | Exponential decay of arrival rate with depth |
| `price_drift` | float | 0.0 | Drift added to mid-price per step |
| `volatility_scale` | float | 1.0 | Multiplier on market order sizes |
| `agent_order_size` | float | 1.0 | Size of agent's limit orders |
| `max_inventory` | int | 50 | Absolute inventory limit |
| `inventory_penalty` | float | 0.001 | Lambda for inventory² penalty |
| `max_steps` | int | 5000 | Maximum episode length |
| `initial_volume_per_level` | float | 5.0 | Starting volume at best level |
| `initial_spread_ticks` | int | 2 | Initial spread in ticks |

### `state.py` — OrderBookState

A `typing.NamedTuple`. The book is **two fixed-size arrays** indexed by distance from mid-price. Index 0 = best bid/ask.

```
bid price at level i = mid_price - (i + half_spread_ticks) * tick_size
ask price at level i = mid_price + (i + half_spread_ticks) * tick_size
```

Fields:
- `bid_volumes: jnp.ndarray` — `(n_levels,)`, float32
- `ask_volumes: jnp.ndarray` — `(n_levels,)`, float32
- `mid_price: jnp.ndarray` — scalar float32
- `half_spread_ticks: jnp.ndarray` — scalar int32
- `regime: jnp.ndarray` — scalar int32 (set to 0 for now, used in Phase 3)
- `inventory: jnp.ndarray` — scalar float32
- `cash: jnp.ndarray` — scalar float32
- `step_count: jnp.ndarray` — scalar int32
- `done: jnp.ndarray` — scalar bool
- `rng_key: jnp.ndarray` — PRNGKey

`init_state(config, rng_key)`: exponentially decaying volume profile `initial_volume_per_level * exp(-depth_decay * i)`. Starting mid_price = 100.0.

### `matching.py` — Matching Engine

**`fill_market_buy(ask_volumes, qty, half_spread_ticks, mid_price, tick_size)`**:
1. `cumsum` of ask_volumes
2. Consumed per level: `consumed[i] = min(vol[i], max(0, qty - cumsum[i-1]))`
3. VWAP fill price from consumed × level_prices
4. Find new best ask with `jnp.argmax(new_volumes > 0)`, shift arrays with `jnp.roll`, zero-fill deep end
5. Update mid-price based on shift
6. Return: `(new_ask_volumes, filled_qty, avg_fill_price, new_mid_price, new_half_spread)`

**`fill_market_sell(...)`**: mirror on bid side.

### `background.py` — Background Order Flow

**`generate_background_flow(bid_volumes, ask_volumes, config, rng_key)`**

For Phase 1, use config parameters directly (no regime_params yet). Split into 10+ subkeys upfront.

1. **Cancellations**: per-level with prob `cancel_prob`, cancel random 0–50% fraction
2. **New limit orders**: rate `limit_order_rate * exp(-depth_decay * i)`, Bernoulli approximation
3. **Market orders**: buy with prob `market_buy_prob`, sell with `market_sell_prob`, size uniform `[min, max]`

Return: `(new_bid_volumes, new_ask_volumes, market_buy_qty, market_sell_qty)`

## Test File: `tests/test_phase1.py`

Run with: `pytest tests/test_phase1.py -v`

```python
"""Phase 1 tests — core book mechanics."""
import pytest
import jax
import jax.numpy as jnp

class TestInitState:
    def test_volumes_are_decreasing(self):
        """Volume at level 0 > volume at level 10 (exponential decay)."""

    def test_mid_price_is_100(self):
        """Starting mid-price should be 100.0."""

    def test_shapes_correct(self):
        """bid_volumes and ask_volumes have shape (n_levels,)."""

    def test_all_fields_are_jax_arrays(self):
        """Every field in OrderBookState must be a jnp.ndarray."""


class TestMatchingEngine:
    def test_partial_fill_single_level(self):
        """Fill qty=3 against ask_volumes=[5, 5, ...]. Level 0: 5→2, filled=3."""
        # Assert filled_qty == 3.0
        # Assert new_ask_volumes[0] == 2.0
        # Assert mid_price unchanged (best level not depleted)

    def test_full_level_depletion(self):
        """Fill qty=5 against ask_volumes=[5, 5, ...]. Level 0 fully consumed."""
        # Assert new_ask_volumes[0] after roll is volume from what was level 1

    def test_multi_level_fill(self):
        """Fill qty=8 against ask_volumes=[5, 5, ...]. Depletes level 0, partially fills level 1."""
        # Assert filled_qty == 8.0
        # Assert volumes correct after roll

    def test_mid_price_shifts_on_depletion(self):
        """When best ask is depleted, mid_price must increase."""
        # Fill enough to deplete level 0
        # Assert new_mid_price > old_mid_price

    def test_vwap_correct(self):
        """VWAP fill price matches hand-calculated value."""
        # ask_volumes = [3, 5, ...], qty = 6
        # Fills 3 at best ask price, 3 at next level price
        # Assert avg_price == weighted average of those two prices

    def test_roll_zero_fill(self):
        """After jnp.roll, deep end must be zeros, not stale wrapped values."""
        # Deplete first 3 levels, check that last 3 levels are 0.0

    def test_sell_mirrors_buy(self):
        """fill_market_sell should be symmetric to fill_market_buy."""
        # Same qty, symmetric book → mid_price moves opposite direction

    def test_zero_qty_noop(self):
        """Filling qty=0 should not change anything."""

    def test_jit_compatible(self):
        """jax.jit(fill_market_buy) runs without error."""


class TestBackgroundFlow:
    def test_market_orders_fire_at_expected_rate(self):
        """Over 1000 calls, market buys should fire ~15% of the time (within 10-20%)."""
        # Run 1000 times with different keys, count how many have market_buy_qty > 0
        # Assert between 100 and 200 out of 1000

    def test_cancellations_reduce_volume(self):
        """After background flow, total volume should generally decrease (cancellations)."""
        # Compare total volume before and after (on average)

    def test_new_limit_orders_add_volume(self):
        """Limit order arrivals add volume, more at near levels than deep."""
        # Check that near levels get more additions than deep levels on average

    def test_rng_keys_independent(self):
        """Different rng keys produce different market order quantities."""
        # Run with key1 and key2, assert results differ

    def test_jit_compatible(self):
        """jax.jit(generate_background_flow) runs without error."""
```

## Success Criteria

All tests pass: `pytest tests/test_phase1.py -v` shows all green.

---

# PHASE 2 — Sim Loop + Sanity Plots

## Scope

Wire everything into a `lax.scan`-compatible step function. No regimes yet — use base config parameters directly. Get mid-price moving and fills happening.

## Files to Create/Modify

- `lob_sim/obs.py`
- `lob_sim/reward.py`
- `lob_sim/step.py`
- `tests/test_phase2.py`
- `scripts/plot_sanity.py`
- Update `lob_sim/__init__.py`

## Specifications

### `obs.py` — Observation Vector

`observe(state, config) → jnp.ndarray` of fixed length `3 * OBS_DEPTH + 3` where `OBS_DEPTH = 10`.

Components concatenated: top-K bid volumes (normalized), top-K ask volumes (normalized), cumulative bid-ask imbalance at each depth `(cum_bid - cum_ask) / (cum_bid + cum_ask + eps)`, spread normalized, inventory normalized, PnL proxy `(cash + inventory * mid_price) / 1000`.

The observation does **NOT** include the regime.

### `reward.py`

`compute_reward(prev_state, next_state, config) → scalar`
```
value(s) = s.cash + s.inventory * s.mid_price
reward = value(next) - value(prev) - config.inventory_penalty * next.inventory²
```

### `step.py`

**`make_step_fn(config) → step_fn`** where `step_fn(state, action) → (state, output_dict)`.

For Phase 2, `action` is a `jnp.ndarray` of shape `(2,)` — raw `[bid_offset_ticks, ask_offset_ticks]`. (Discrete action table comes in Phase 3.)

Step logic:
1. Place agent orders at offset levels, add volume
2. Snapshot pre-match volumes at agent levels
3. Generate background flow (using config params directly)
4. Match market orders
5. Determine agent fills (pro-rata: `consumed / pre_match_vol * agent_size`, with div-by-zero guard)
6. Remove unfilled agent volume from book
7. Compute obs, reward
8. Build output dict with: obs, reward, mid_price, spread, inventory, cash, bid_fill, ask_fill, done, bid_volumes[:20], ask_volumes[:20]
9. Done if `abs(inventory) >= max_inventory` or `step_count >= max_steps`

**`run_episode(config, rng_key, actions)`**: `actions` shape `(T, 2)`. Uses `lax.scan`.

## Test File: `tests/test_phase2.py`

Run with: `pytest tests/test_phase2.py -v`

```python
"""Phase 2 tests — sim loop, obs, reward."""
import pytest
import jax
import jax.numpy as jnp

T = 2000  # steps for integration tests

class TestObservation:
    def test_output_shape(self):
        """observe() returns array of shape (obs_size(),)."""

    def test_obs_size_matches(self):
        """obs_size() returns 3 * 10 + 3 = 33."""

    def test_no_nan(self):
        """Observation should have no NaN values for a fresh state."""

    def test_jit_compatible(self):
        """jax.jit(observe) works."""


class TestReward:
    def test_zero_when_nothing_changes(self):
        """If state doesn't change, reward should be ~0 (minus small inventory penalty)."""

    def test_positive_when_profitable(self):
        """Reward positive when portfolio value increases."""

    def test_inventory_penalty(self):
        """Higher inventory → more negative penalty component."""


class TestStepFunction:
    def test_jit_compiles(self):
        """jax.jit(step_fn) runs without tracing error."""

    def test_step_count_increments(self):
        """After one step, step_count should be 1."""

    def test_agent_orders_removed_after_step(self):
        """Unfilled agent volume should not accumulate in the book."""
        # Run two steps, check that book volume doesn't grow unboundedly

    def test_done_on_max_inventory(self):
        """Done flag triggers when abs(inventory) >= max_inventory."""


class TestRunEpisode:
    def test_lax_scan_compiles(self):
        """run_episode with T steps compiles under jax.jit."""
        # actions = jnp.broadcast_to(jnp.array([3.0, 3.0]), (T, 2))
        # jax.jit(run_episode)(config, key, actions)

    def test_mid_price_not_flat(self):
        """Mid-price trajectory must NOT be constant — this is the #1 sanity check."""
        # Run T=2000 steps
        # Assert std(mid_prices) > 0.001

    def test_fills_happen(self):
        """At least some bid or ask fills should occur over T steps."""
        # Assert sum(bid_fills) > 0 or sum(ask_fills) > 0

    def test_inventory_fluctuates(self):
        """Inventory should change over time (not stuck at 0)."""
        # Assert max(abs(inventory)) > 0

    def test_output_shapes(self):
        """All output arrays have leading dim T."""

    def test_throughput(self):
        """Warm throughput > 5,000 steps/sec single env."""
        # Time the second call (after JIT compilation)
        # Assert T / elapsed > 5000
```

## Plot File: `scripts/plot_sanity.py`

Run with: `python scripts/plot_sanity.py`

Generates `plots/sanity.png` — a 3×2 matplotlib figure:
1. Mid-price trajectory
2. Spread over time
3. LOB depth snapshot at T/2 (bids left green, asks right red)
4. LOB depth heatmap (x=time, y=level, color=volume)
5. Agent inventory
6. Cumulative reward

Prints summary stats: price range, std, mean spread, final inventory, total reward, compile time, warm steps/sec.

**This is for visual inspection only — the pytest file is the pass/fail gate.**

## Success Criteria

All tests pass: `pytest tests/test_phase2.py -v` shows all green. Then visually inspect `scripts/plot_sanity.py` output to confirm plots look reasonable.

---

# PHASE 3 — Regimes + Discrete Actions

## Scope

Add the HMM regime system and the 5×5 discrete action table. Refactor background flow to be regime-conditioned. Add locked-regime mode.

## Files to Create/Modify

- `lob_sim/regime.py` (new)
- `lob_sim/actions.py` (new)
- `lob_sim/background.py` (modify: add `regime_params` argument)
- `lob_sim/step.py` (modify: regime transition, discrete actions, `locked_regime`)
- `tests/test_phase3.py` (new)
- `scripts/plot_regimes.py` (new)

## Specifications

### `regime.py`

Constants: `NOISE=0, BULL=1, BEAR=2, N_REGIMES=3`

Transition matrix:
```
NOISE: [0.98, 0.01, 0.01]
BULL:  [0.02, 0.97, 0.01]
BEAR:  [0.02, 0.01, 0.97]
```

Per-regime parameter arrays (each shape `(3,)` indexed by regime):

| Parameter | Noise | Bull | Bear |
|---|---|---|---|
| `market_buy_prob` | 0.15 | 0.25 | 0.08 |
| `market_sell_prob` | 0.15 | 0.08 | 0.25 |
| `price_drift` | 0.0 | +0.002 | -0.002 |
| `volatility_scale` | 1.0 | 1.3 | 1.3 |
| `cancel_prob` | 0.02 | 0.04 | 0.04 |
| `limit_order_rate` | 0.5 | 0.4 | 0.4 |

`RegimeStepParams` NamedTuple with these 6 fields.
`transition_regime(current, rng_key) → new_regime` via `jax.random.choice`.
`get_regime_params(regime) → RegimeStepParams` via array indexing.

### `actions.py`

`BID_TICKS = [1,2,3,4,5]`, `ASK_TICKS = [1,2,3,4,5]` → 25 actions.
`ACTION_TABLE = jnp.array(list(itertools.product(...)))` shape `(25, 2)`.
`N_ACTIONS = 25`.
Helper functions: `action_index_to_offsets(idx)`, `offsets_to_action_index(bid, ask)`.

### Modifications to `step.py`

- `make_step_fn(config, locked_regime=-1)` — locked_regime is closed over
- Action is now scalar int32 (index 0–24), looked up via `ACTION_TABLE[action]`
- Step 1: transition regime, apply `locked_regime` override via `jnp.where`, get regime params
- Step 2: apply `mid_price += regime_params.price_drift`
- Background flow receives `regime_params` instead of config for varying params
- Output dict includes `"regime": new_regime`
- `run_episode(config, rng_key, actions, locked_regime=-1)` — actions shape `(T,)` int32

### Modifications to `background.py`

New signature: `generate_background_flow(bid_volumes, ask_volumes, regime_params, config, rng_key)`
Market order sizes multiplied by `regime_params.volatility_scale`.

## Test File: `tests/test_phase3.py`

Run with: `pytest tests/test_phase3.py -v`

```python
"""Phase 3 tests — regimes, discrete actions, locked-regime mode."""
import pytest
import jax
import jax.numpy as jnp

class TestRegime:
    def test_transition_matrix_rows_sum_to_one(self):
        """Each row of TRANSITION_MATRIX sums to 1.0."""

    def test_transition_stays_mostly(self):
        """Over 1000 transitions from NOISE, >90% should stay NOISE."""

    def test_all_regimes_reachable(self):
        """Over 10000 transitions from NOISE, all 3 regimes should appear."""

    def test_get_regime_params_noise(self):
        """get_regime_params(NOISE) returns market_buy_prob=0.15."""

    def test_get_regime_params_bull(self):
        """get_regime_params(BULL) returns market_buy_prob=0.25, market_sell_prob=0.08."""

    def test_jit_compatible(self):
        """transition_regime and get_regime_params work under jit."""


class TestActions:
    def test_table_shape(self):
        """ACTION_TABLE has shape (25, 2)."""

    def test_table_contents(self):
        """ACTION_TABLE[0] = [1,1], ACTION_TABLE[24] = [5,5]."""

    def test_round_trip(self):
        """offsets_to_action_index(action_index_to_offsets(i)) == i for all i."""


class TestLockedRegime:
    def test_locked_noise_stays_noise(self):
        """With locked_regime=0, regime is 0 for all T steps."""
        # Run T=500 steps, assert all output regimes == 0

    def test_locked_bull_stays_bull(self):
        """With locked_regime=1, regime is 1 for all T steps."""

    def test_unlocked_switches(self):
        """With locked_regime=-1, at least 2 different regimes appear over T=5000 steps."""


class TestRegimeEffectOnDynamics:
    def test_bull_drifts_up(self):
        """In locked bull regime, mean mid-price change > 0 over T=2000 steps."""
        # Assert mid_prices[-1] > mid_prices[0] (with high probability)

    def test_bear_drifts_down(self):
        """In locked bear regime, mean mid-price change < 0 over T=2000 steps."""

    def test_noise_no_drift(self):
        """In locked noise regime, abs(mid_prices[-1] - mid_prices[0]) < bull drift."""

    def test_bull_more_ask_fills(self):
        """In bull regime, total ask fills > total bid fills (more market buys)."""
        # Run T=2000 with symmetric agent, compare sum of fills

    def test_bear_more_bid_fills(self):
        """In bear regime, total bid fills > total ask fills."""

    def test_run_episode_with_discrete_action(self):
        """run_episode works with scalar int32 actions."""
        # actions = jnp.full((T,), 12, dtype=jnp.int32)  # action 12 = (3,3)
        # Should compile and run
```

## Plot File: `scripts/plot_regimes.py`

Generates `plots/regimes.png` — 2×2 figure:
1. Mid-price trajectories overlaid (3 colors, one per locked regime)
2. Cumulative bid fills vs ask fills per regime
3. Inventory paths per regime
4. Average LOB depth per regime

## Success Criteria

All tests pass: `pytest tests/test_phase3.py -v` shows all green. Previous phases still pass: `pytest tests/test_phase1.py tests/test_phase2.py -v`.

---

# PHASE 4 — Policy Divergence Proof

## Scope

Monte Carlo evaluation proving that different regimes have different optimal actions. This validates the entire research premise.

## Files to Create

- `scripts/plot_divergence.py`
- `tests/test_phase4.py`

## Specification

For each of 25 actions × 3 regimes: run N=200 episodes of T=1000 steps (locked regime, fixed action). Use `jax.vmap` over episodes.

```python
keys = jax.random.split(master_key, N)
actions_repeated = jnp.full((T,), action_idx, dtype=jnp.int32)
batched_run = jax.vmap(lambda k: run_episode(config, k, actions_repeated, locked_regime=regime))
_, outputs = batched_run(keys)
mean_reward = outputs["reward"].sum(axis=1).mean()
```

## Test File: `tests/test_phase4.py`

Run with: `pytest tests/test_phase4.py -v`

This test is heavier — it runs the full Monte Carlo evaluation. Use `pytest -v --timeout=300` if timeout is configured.

```python
"""Phase 4 tests — policy divergence across regimes."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Use smaller N for faster tests, but enough to be statistically meaningful
N_EPISODES = 100  # reduced from 200 for test speed
T_STEPS = 500     # reduced from 1000 for test speed

@pytest.fixture(scope="module")
def reward_matrix():
    """Compute the (25, 3) reward matrix once for all tests.
    reward_matrix[action_idx, regime_idx] = mean total reward.
    """
    # ... compute and return

class TestPolicyDivergence:
    def test_completes_in_reasonable_time(self, reward_matrix):
        """The full evaluation should complete (this test just checks it ran)."""
        assert reward_matrix.shape == (25, 3)

    def test_optimal_actions_differ(self, reward_matrix):
        """Optimal action (argmax) is different for each regime."""
        optimal = reward_matrix.argmax(axis=0)  # shape (3,)
        # All 3 optimal actions should be different
        assert len(set(optimal.tolist())) == 3, (
            f"Optimal actions are not all different: {optimal}. "
            f"Regime params may need more asymmetry."
        )

    def test_noise_optimal_is_symmetric(self, reward_matrix):
        """Noise regime optimal action should have bid_tick ≈ ask_tick."""
        optimal_idx = reward_matrix[:, 0].argmax()
        bid, ask = ACTION_TABLE[optimal_idx]
        assert abs(int(bid) - int(ask)) <= 1, (
            f"Noise optimal ({bid},{ask}) is not symmetric."
        )

    def test_bull_optimal_tight_bid_wide_ask(self, reward_matrix):
        """Bull optimal should have bid_tick < ask_tick."""
        optimal_idx = reward_matrix[:, 1].argmax()
        bid, ask = ACTION_TABLE[optimal_idx]
        assert int(bid) < int(ask), (
            f"Bull optimal ({bid},{ask}) should have bid < ask."
        )

    def test_bear_optimal_wide_bid_tight_ask(self, reward_matrix):
        """Bear optimal should have bid_tick > ask_tick."""
        optimal_idx = reward_matrix[:, 2].argmax()
        bid, ask = ACTION_TABLE[optimal_idx]
        assert int(bid) > int(ask), (
            f"Bear optimal ({bid},{ask}) should have bid > ask."
        )

    def test_cross_regime_penalty(self, reward_matrix):
        """Using the wrong regime's optimal action should cost >10% reward."""
        for regime in range(3):
            best_action = reward_matrix[:, regime].argmax()
            best_reward = reward_matrix[best_action, regime]
            for other_regime in range(3):
                if other_regime == regime:
                    continue
                other_best = reward_matrix[:, other_regime].argmax()
                wrong_reward = reward_matrix[other_best, regime]
                # Wrong policy should be worse
                assert wrong_reward < best_reward, (
                    f"Wrong policy (regime {other_regime}'s optimal) applied in "
                    f"regime {regime} is not worse: {wrong_reward} vs {best_reward}"
                )
```

## Plot File: `scripts/plot_divergence.py`

Runs the full N=200, T=1000 evaluation. Generates `plots/divergence.png`:
1. Three 5×5 heatmaps (shared color scale)
2. Prints optimal action per regime
3. Cross-regime penalty table
4. Cohen's d effect sizes

## Success Criteria

All tests pass: `pytest tests/test_phase4.py -v`. If `test_optimal_actions_differ` fails, increase regime asymmetry (bull `market_buy_prob` → 0.30, `market_sell_prob` → 0.05) and re-run.

---

# PHASE 5 — Performance Benchmarking

## Scope

Benchmark throughput with `vmap` batching. Verify the sim is fast enough for RL training.

## Files to Create

- `tests/test_phase5.py`

## Test File: `tests/test_phase5.py`

```python
"""Phase 5 tests — performance benchmarks."""
import pytest
import jax
import jax.numpy as jnp
import time

T = 5000

class TestPerformance:
    def test_single_env_throughput(self):
        """Warm single-env throughput > 10,000 steps/sec."""
        # Compile
        run_jit = jax.jit(lambda k, a: run_episode(config, k, a, locked_regime=-1))
        actions = jnp.full((T,), 12, dtype=jnp.int32)
        key = jax.random.PRNGKey(0)
        run_jit(key, actions)  # compile
        jax.block_until_ready(run_jit(key, actions))  # ensure ready
        # Time
        t0 = time.perf_counter()
        _, out = run_jit(key, actions)
        jax.block_until_ready(out["mid_price"])
        elapsed = time.perf_counter() - t0
        steps_per_sec = T / elapsed
        print(f"Single env: {steps_per_sec:.0f} steps/sec")
        assert steps_per_sec > 10_000

    def test_vmap_compiles(self):
        """vmap over 64 envs compiles without error."""
        batch_size = 64
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        actions = jnp.full((T,), 12, dtype=jnp.int32)
        batched = jax.jit(jax.vmap(
            lambda k: run_episode(config, k, actions, locked_regime=-1)
        ))
        _, out = batched(keys)
        jax.block_until_ready(out["mid_price"])
        assert out["mid_price"].shape == (batch_size, T)

    def test_vmap_scaling(self):
        """Throughput scales with batch size."""
        results = {}
        actions = jnp.full((T,), 12, dtype=jnp.int32)
        for batch_size in [1, 16, 64, 256]:
            keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
            batched = jax.jit(jax.vmap(
                lambda k: run_episode(config, k, actions, locked_regime=-1)
            ))
            batched(keys)  # compile
            jax.block_until_ready(batched(keys))
            t0 = time.perf_counter()
            _, out = batched(keys)
            jax.block_until_ready(out["mid_price"])
            elapsed = time.perf_counter() - t0
            total_steps = batch_size * T
            results[batch_size] = total_steps / elapsed
            print(f"Batch {batch_size}: {results[batch_size]:.0f} total steps/sec")
        # Batch 64 should be faster total throughput than batch 1
        assert results[64] > results[1] * 2

    def test_no_oom_batch_1024(self):
        """Batch size 1024 doesn't OOM (may be slow, just verify it works)."""
        keys = jax.random.split(jax.random.PRNGKey(0), 1024)
        actions = jnp.full((1000,), 12, dtype=jnp.int32)  # shorter T
        batched = jax.jit(jax.vmap(
            lambda k: run_episode(config, k, actions, locked_regime=-1)
        ))
        _, out = batched(keys)
        jax.block_until_ready(out["mid_price"])
        assert out["mid_price"].shape == (1024, 1000)
```

## Success Criteria

All tests pass: `pytest tests/test_phase5.py -v -s` (with `-s` to see throughput numbers).

---

# PHASE 6 — Agent Framework + PPO Baseline

## Scope

Build the agent interface, rollout collection, GAE computation, PPO training loop, and the vanilla PPO agent. Train PPO on a single locked regime as proof of concept.

## Additional Dependencies

Add to pyproject.toml: `optax`, `distrax`, `equinox`

## Files to Create

- `lob_sim/agents/__init__.py`
- `lob_sim/agents/base.py`
- `lob_sim/agents/networks.py`
- `lob_sim/agents/ppo.py`
- `lob_sim/training/__init__.py`
- `lob_sim/training/rollout.py`
- `lob_sim/training/trainer.py`
- `lob_sim/training/eval.py`
- `lob_sim/training/logger.py`
- `scripts/train.py`
- `tests/test_phase6.py`

## Specifications

### `agents/base.py`

```python
class Agent(eqx.Module):
    def initial_agent_state(self, rng_key) -> AgentState: ...
    def get_action(self, obs, agent_state, rng_key) -> tuple[action, new_state, info]: ...
    def update_policy(self, rollout_data, opt_state) -> tuple[Agent, opt_state, metrics]: ...
```

`info` must include `{"log_prob": scalar, "value": scalar}`.

### `agents/networks.py`

Equinox: `MLP(in_size, hidden_sizes, out_size, key)`, `GRUCell`.

### `agents/ppo.py`

`obs (33,) → MLP [64,64] → policy_head (25,) → Categorical; → value_head (1,)`.

PPOConfig: lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, n_epochs=4, n_minibatches=4, n_envs=64, n_steps=256.

### `training/rollout.py`

Closed-loop via `lax.scan` — agent picks actions inside the loop. `vmap` over N envs.

### `training/trainer.py`

GAE via reverse `lax.scan`. PPO update loop: n_epochs × n_minibatches.

### `scripts/train.py`

```
python scripts/train.py --agent ppo --regime noise --n_iterations 500 --seed 42
```

## Test File: `tests/test_phase6.py`

```python
"""Phase 6 tests — agent framework + PPO training."""
import pytest
import jax

class TestAgentInterface:
    def test_ppo_initial_state(self):
        """PPO agent creates a valid initial AgentState."""

    def test_ppo_get_action_shape(self):
        """get_action returns (scalar_action, agent_state, info_dict)."""
        # action should be scalar int32 in [0, 24]
        # info should have 'log_prob' and 'value' keys

    def test_ppo_get_action_jit(self):
        """get_action is JIT-compatible."""


class TestRollout:
    def test_rollout_compiles(self):
        """collect_rollout with PPO agent compiles under jit."""

    def test_rollout_shapes(self):
        """Rollout data has correct shapes (n_steps, ...)."""
        # obs: (n_steps, 33), action: (n_steps,), reward: (n_steps,), etc.

    def test_batched_rollout(self):
        """vmap over n_envs produces (n_envs, n_steps, ...) shaped data."""

    def test_actions_in_valid_range(self):
        """All actions in rollout are in [0, 24]."""


class TestGAE:
    def test_gae_shape(self):
        """compute_gae returns advantages and returns with shape (n_steps,)."""

    def test_gae_known_case(self):
        """GAE matches hand-computed value for a simple 3-step trajectory."""
        # rewards = [1, 0, 1], values = [0.5, 0.5, 0.5], dones = [0, 0, 1]
        # Verify advantages match expected values


class TestPPOTraining:
    def test_single_update(self):
        """One PPO update step runs without error and returns metrics."""
        # metrics should include 'policy_loss', 'value_loss', 'entropy'

    def test_loss_decreases_over_updates(self):
        """After 50 training iterations on locked noise, mean reward improves."""
        # Train for 50 iterations
        # Compare mean reward: last 10 iterations vs first 10
        # Assert improvement (last > first, or at least not dramatically worse)

    def test_action_distribution_shifts(self):
        """After training on noise, agent should prefer symmetric actions."""
        # Evaluate trained agent, collect action counts
        # Most common action should be roughly symmetric (bid ≈ ask)
```

## Success Criteria

All tests pass: `pytest tests/test_phase6.py -v`. PPO trains and reward improves.

---

# PHASE 7 — RL² Agent

## Scope

Add the RL² recurrent baseline. GRU hidden state carries across the entire meta-episode.

## Files to Create

- `lob_sim/agents/rl2.py`
- `tests/test_phase7.py`
- Modify `training/rollout.py` if needed for meta-episode support

## Specification

```
[obs (33,) ; prev_action (25,) one-hot ; prev_reward (1,)] → GRU (hidden=128) → hidden
hidden → policy_head (25,) / value_head (1,)
```

`RL2AgentState`: GRU hidden `(128,)`, prev_action `int32`, prev_reward `float32`.

Meta-episode: `n_episodes_per_meta_episode = 3`. Environment resets mid-rollout, agent hidden state persists. Pass `done` flag as additional GRU input.

## Test File: `tests/test_phase7.py`

```python
"""Phase 7 tests — RL² agent."""
import pytest
import jax

class TestRL2Agent:
    def test_initial_state_shape(self):
        """Hidden state shape is (128,)."""

    def test_get_action_updates_hidden(self):
        """Hidden state changes after get_action call."""
        # Call get_action twice with same obs, verify hidden states differ

    def test_input_includes_prev_action_reward(self):
        """GRU input is [obs, prev_action_onehot, prev_reward] = dim 33+25+1 = 59."""

    def test_jit_compatible(self):
        """get_action compiles under jit."""

    def test_hidden_persists_across_done(self):
        """When done=True, hidden state is NOT zeroed in meta-episode mode."""


class TestRL2Training:
    def test_trains_without_error(self):
        """50 iterations of training on mixed regimes completes."""

    def test_outperforms_ppo(self):
        """After training, RL² mean reward > PPO mean reward on mixed regimes."""
        # Train both for same number of iterations
        # Evaluate both on same set of episodes
        # RL² should be higher (or at least comparable)

    def test_hidden_state_correlates_with_regime(self):
        """In evaluation, cluster hidden states — should correlate with true regime."""
        # Run evaluation, collect (hidden_state, true_regime) pairs
        # Simple test: mean hidden state norm differs across regimes
        # (not a hard pass/fail — print the values for inspection)
```

## Success Criteria

All tests pass: `pytest tests/test_phase7.py -v`.

---

# PHASE 8 — VariBAD

## Scope

Implement VariBAD (Zintgraf et al., 2020): VAE encoder-decoder with belief-conditioned policy.

## Files to Create

- `lob_sim/agents/varibad.py`
- `lob_sim/training/vae_buffer.py`
- `tests/test_phase8.py`
- Modify `training/trainer.py` for dual-optimizer training

## Specification

### Encoder

`(sₜ, aₜ₋₁, rₜ) → FC layers → GRU (hidden=64) → linear → (μ, log_σ) ∈ ℝ^5 each`

### Decoder

`(sₜ, aₜ, m) → MLP [64, 32] → predicted r_{t+1}`

### Policy

`[obs (33,) ; μ (5,) ; σ (5,)] → MLP [64, 64] → policy_head (25,) / value_head (1,)`

### Training

Separate optimizers: PPO (lr=7e-4) for policy, Adam (lr=1e-3) for VAE. RL loss does NOT backprop through encoder. VAE trains from trajectory replay buffer. ELBO at 8 subsampled timesteps per trajectory.

## Test File: `tests/test_phase8.py`

```python
"""Phase 8 tests — VariBAD."""
import pytest
import jax
import jax.numpy as jnp

class TestVariBADAgent:
    def test_initial_posterior_is_prior(self):
        """At t=0, posterior should be N(0,I): μ≈0, σ≈1."""

    def test_posterior_updates(self):
        """After processing observations, μ and σ should change from prior."""
        # Run a few steps, verify posterior != prior

    def test_policy_input_includes_posterior(self):
        """Policy receives [obs, μ, σ] = dim 33+5+5 = 43."""

    def test_get_action_jit(self):
        """get_action compiles under jit."""

    def test_reparameterization(self):
        """Latent samples via reparameterization: m = μ + σ * ε."""


class TestVAEBuffer:
    def test_add_and_sample(self):
        """Can add trajectories and sample batches."""

    def test_max_size(self):
        """Buffer doesn't grow beyond max_trajectories."""


class TestVariBADTraining:
    def test_vae_reconstruction_loss_decreases(self):
        """After 100 iterations, reward decoder loss is lower than at start."""
        # Record loss at iteration 0 and iteration 100
        # Assert iteration_100_loss < iteration_0_loss

    def test_kl_not_collapsed(self):
        """KL divergence should be > 0.1 (latent is being used)."""
        # If KL ≈ 0, the model is ignoring the latent → posterior collapse

    def test_latent_clusters_by_regime(self):
        """Latent embeddings should cluster by true regime."""
        # Run evaluation episodes, collect (μ, true_regime) pairs
        # Compute mean μ per regime
        # Assert that mean μ differs across regimes (e.g., L2 distance > threshold)
        means = []  # mean μ for each regime
        for regime in [0, 1, 2]:
            # ... collect μ values when true regime == regime
            means.append(mu_mean)
        # All pairwise distances should be > 0
        for i in range(3):
            for j in range(i+1, 3):
                dist = jnp.linalg.norm(means[i] - means[j])
                assert dist > 0.1, f"Regimes {i} and {j} not separated: dist={dist}"

    def test_outperforms_ppo(self):
        """After training, VariBAD mean reward > PPO on mixed regimes."""

    def test_separate_optimizers(self):
        """Policy optimizer and VAE optimizer are separate (different lr, different params)."""
```

## Success Criteria

All tests pass: `pytest tests/test_phase8.py -v`.

---

# PHASE 9 — VariBAD + HyperNetwork

## Scope

Add the hypernetwork extension (Beck et al., 2022, 2023).

## Files to Create

- `lob_sim/agents/varibad_hyper.py`
- Add `HyperNetwork` to `agents/networks.py`
- `tests/test_phase9.py`

## Specification

### HyperNetwork

`m ∈ ℝ^5 → ReLU → h(m) = generated policy weights φ`

Separate heads for actor and critic. Linear hypernetwork (single layer).

### Bias-HyperInit

1. Init base network `f` with normc initialization
2. hypernetwork.bias = flatten(f.parameters())
3. hypernetwork.weight = small random (near-zero)

So `h(m) ≈ f` for any `m` at init.

## Test File: `tests/test_phase9.py`

```python
"""Phase 9 tests — VariBAD + HyperNetwork."""
import pytest
import jax
import jax.numpy as jnp

class TestHyperNetwork:
    def test_generates_correct_shapes(self):
        """HyperNetwork output matches base network parameter shapes."""

    def test_bias_hyperinit(self):
        """At init, h(random_m) ≈ base_network params (small difference)."""
        # Generate weights for random m
        # Compare to base network init
        # Assert max difference < 0.1

    def test_different_m_different_weights(self):
        """Different latent m produces different policy weights."""
        # h(m1) != h(m2) for m1 != m2


class TestVariBADHyperAgent:
    def test_get_action_jit(self):
        """get_action compiles under jit."""

    def test_generated_policy_works(self):
        """Generated policy produces valid action distributions."""
        # Softmax of policy logits sums to 1, all > 0


class TestVariBADHyperTraining:
    def test_trains_without_divergence(self):
        """100 training iterations complete with finite loss values."""
        # Assert no NaN in any loss

    def test_matches_or_beats_varibad(self):
        """VariBAD+HN mean reward ≥ 0.9 * VariBAD mean reward on mixed regimes."""
        # Allow some tolerance — main check is it doesn't collapse

    def test_weight_distance_across_regimes(self):
        """Generated weights differ when conditioned on different regime embeddings."""
        # Get μ for each regime (from locked-regime eval)
        # Generate weights for each μ
        # Assert L2 distance between weight vectors > threshold
```

## Success Criteria

All tests pass: `pytest tests/test_phase9.py -v`.

---

# PHASE 10 — Comparison Experiments + Thesis Analysis

## Scope

Run all agents head-to-head with full thesis-quality rigor: multiple seeds, statistical tests, oracle baseline, regime persistence ablation, learning curves, and latent space analysis.

## Files to Create

- `lob_sim/agents/oracle.py`
- `scripts/train_all_seeds.py`
- `scripts/eval_agent.py`
- `scripts/compare_agents.py`
- `scripts/ablation_persistence.py`
- `tests/test_phase10.py`

## Specifications

### Oracle Agent (`agents/oracle.py`)

A "cheating" agent that receives the **true regime** as input and plays the Monte Carlo optimal action for that regime (from Phase 4's `verify_policy_divergence.py` results). This is the theoretical performance ceiling — no learned agent can beat it.

```python
class OracleAgent(eqx.Module):
    optimal_actions: jnp.ndarray  # shape (3,) — best action index per regime

    def get_action(self, obs, agent_state, rng_key):
        # Cheats: reads true regime from env state (passed via agent_state)
        return self.optimal_actions[true_regime], agent_state, info
```

This anchors all comparison plots: oracle at top, PPO at bottom, learned agents in between.

### Multi-Seed Training (`scripts/train_all_seeds.py`)

Train each agent (PPO, RL², VariBAD, VariBAD+HN) with 5 seeds (42–46) on mixed regimes. This is the same as `train.py` but loops over seeds and agents.

```
python scripts/train_all_seeds.py --n_iterations 1000 --n_envs 64 --save_dir checkpoints/
```

Produces: `checkpoints/{agent}_{seed}.eqx` — 20 checkpoints total (4 agents × 5 seeds).

### Comparison Script (`scripts/compare_agents.py`)

Loads all 20 checkpoints. For each agent, evaluates all 5 seeds on N=200 mixed-regime episodes. Reports mean ± std across seeds.

Generates `plots/comparison.png` — multi-panel figure:

1. **Bar chart with error bars**: mean total reward per agent ± std across 5 seeds. Include oracle as a horizontal dashed line at the top. Order: Oracle, VariBAD+HN, VariBAD, RL², PPO.

2. **Per-regime breakdown**: grouped bar chart — for each regime, show all agents' mean reward. This reveals which agents adapt per regime vs play a compromise.

3. **Adaptation speed**: for each agent, compute the average windowed reward (window=20 steps) in the 50 steps before and 50 steps after every regime transition in the evaluation episodes. Plot these as curves (x = steps relative to transition, y = mean reward). VariBAD should recover faster.

4. **Learning curves**: mean evaluation reward vs training iteration, one line per agent, shaded region = std across seeds. Evaluated every 50 training iterations. Shows both convergence speed and final performance.

5. **VariBAD latent space**: 2D PCA of latent embeddings (μ) collected during evaluation, colored by true regime. Should show 3 clusters.

6. **Results table** printed to stdout:
   ```
   Agent          | Mean Reward ± Std | Noise  | Bull   | Bear   | p vs PPO
   Oracle         | X.XX ± 0.00       | X.XX   | X.XX   | X.XX   | —
   VariBAD+HN     | X.XX ± X.XX       | X.XX   | X.XX   | X.XX   | 0.XXX
   VariBAD        | X.XX ± X.XX       | X.XX   | X.XX   | X.XX   | 0.XXX
   RL²            | X.XX ± X.XX       | X.XX   | X.XX   | X.XX   | 0.XXX
   PPO            | X.XX ± X.XX       | X.XX   | X.XX   | X.XX   | —
   ```

### Statistical Tests

For each agent pair comparison, use a **paired Wilcoxon signed-rank test** on per-episode rewards (paired by episode RNG seed). Report p-values in the results table. Use `scipy.stats.wilcoxon`. A result is significant at p < 0.05.

Also report **Cohen's d** between each agent and PPO baseline, as a standardized effect size.

### Regime Persistence Ablation (`scripts/ablation_persistence.py`)

Test how VariBAD's advantage depends on regime switching speed. Run the comparison at 3 transition matrix settings:

| Setting | Self-transition prob | Expected duration |
|---|---|---|
| Fast | 0.90 | ~10 steps |
| Medium | 0.97 | ~33 steps (default) |
| Slow | 0.995 | ~200 steps |

For each setting, evaluate all 4 trained agents (using the medium-trained checkpoints — don't retrain). Plot VariBAD's advantage over PPO (reward difference) as a function of regime persistence.

Expected: advantage is small for fast switching (too noisy to infer), peaks at medium persistence, and decreases for slow switching (even PPO adapts eventually).

Generates `plots/ablation_persistence.png`.

### Latent Regime Inference Analysis

For VariBAD, track how well the learned latent tracks the true regime over the course of training:

1. Every 50 training iterations, run a short evaluation (50 episodes) and collect `(μ, true_regime)` pairs
2. Train a simple 3-class logistic regression on μ → regime (using sklearn or manual)
3. Record classification accuracy
4. Plot accuracy vs training iteration

This shows *when* VariBAD discovers the regime structure. Generates `plots/latent_inference_accuracy.png`.

## Test File: `tests/test_phase10.py`

```python
"""Phase 10 tests — final comparison and thesis validation."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

# Assumes all agents trained with 5 seeds and saved as checkpoints
N_EVAL = 200  # evaluation episodes per seed
N_SEEDS = 5

@pytest.fixture(scope="module")
def eval_results():
    """Evaluate all agents on mixed-regime episodes across 5 seeds.
    Returns dict: {agent_name: {"rewards": (N_SEEDS, N_EVAL), "per_regime": ..., ...}}
    """
    # Load all checkpoints, evaluate, return structured results

@pytest.fixture(scope="module")
def oracle_reward(eval_results):
    """Oracle agent's mean reward (theoretical ceiling)."""
    return eval_results["oracle"]["rewards"].mean()


class TestAgentRanking:
    def test_varibad_beats_ppo(self, eval_results):
        """VariBAD mean reward > PPO mean reward (across seeds)."""
        vb = eval_results["varibad"]["rewards"].mean(axis=1)   # (N_SEEDS,)
        ppo = eval_results["ppo"]["rewards"].mean(axis=1)
        assert vb.mean() > ppo.mean(), (
            f"VariBAD ({vb.mean():.4f}) did not beat PPO ({ppo.mean():.4f})"
        )

    def test_varibad_beats_ppo_significant(self, eval_results):
        """Wilcoxon test: VariBAD vs PPO is statistically significant (p < 0.05)."""
        vb = eval_results["varibad"]["rewards"].mean(axis=1)
        ppo = eval_results["ppo"]["rewards"].mean(axis=1)
        stat, p = stats.wilcoxon(vb, ppo, alternative="greater")
        assert p < 0.05, f"VariBAD vs PPO not significant: p={p:.4f}"

    def test_varibad_hyper_beats_ppo(self, eval_results):
        """VariBAD+HN mean reward > PPO mean reward."""
        vbh = eval_results["varibad_hyper"]["rewards"].mean(axis=1)
        ppo = eval_results["ppo"]["rewards"].mean(axis=1)
        assert vbh.mean() > ppo.mean()

    def test_rl2_beats_ppo(self, eval_results):
        """RL² mean reward > PPO mean reward."""
        rl2 = eval_results["rl2"]["rewards"].mean(axis=1)
        ppo = eval_results["ppo"]["rewards"].mean(axis=1)
        assert rl2.mean() > ppo.mean()

    def test_all_below_oracle(self, eval_results, oracle_reward):
        """No learned agent should exceed the oracle (sanity check)."""
        for agent in ["ppo", "rl2", "varibad", "varibad_hyper"]:
            agent_mean = eval_results[agent]["rewards"].mean()
            assert agent_mean <= oracle_reward * 1.05, (  # 5% tolerance for noise
                f"{agent} ({agent_mean:.4f}) exceeds oracle ({oracle_reward:.4f})"
            )


class TestPolicySpecialization:
    def test_ppo_uses_compromise_policy(self, eval_results):
        """PPO most-common action should be similar across regimes."""
        actions_by_regime = eval_results["ppo"]["actions_by_regime"]  # {regime: array}
        modes = [np.bincount(actions_by_regime[r]).argmax() for r in [0, 1, 2]]
        # PPO should use roughly the same action everywhere
        assert len(set(modes)) <= 2, f"PPO uses {len(set(modes))} different actions — expected compromise"

    def test_varibad_specializes_per_regime(self, eval_results):
        """VariBAD most-common action should differ between bull and bear."""
        actions = eval_results["varibad"]["actions_by_regime"]
        bull_mode = np.bincount(actions[1]).argmax()
        bear_mode = np.bincount(actions[2]).argmax()
        assert bull_mode != bear_mode, (
            f"VariBAD uses same action in bull and bear: {bull_mode}"
        )

    def test_varibad_matches_monte_carlo_optimal(self, eval_results):
        """VariBAD's preferred action per regime matches Phase 4 optimal within top-3."""
        # Load Phase 4 reward_matrix, get top-3 actions per regime
        # Check that VariBAD's most-common action is in the top-3


class TestAdaptationSpeed:
    def test_varibad_adapts_faster_than_ppo(self, eval_results):
        """In first 20 steps after regime switch, VariBAD reward > PPO reward."""
        vb_post = eval_results["varibad"]["post_switch_reward"]  # mean reward steps 1-20 after switch
        ppo_post = eval_results["ppo"]["post_switch_reward"]
        assert vb_post > ppo_post, (
            f"VariBAD post-switch reward ({vb_post:.4f}) not higher than PPO ({ppo_post:.4f})"
        )


class TestLatentRepresentation:
    def test_latent_clusters_by_regime(self, eval_results):
        """VariBAD latents should be classifiable by regime with >70% accuracy."""
        mu = eval_results["varibad"]["latent_mu"]       # (N, 5)
        regimes = eval_results["varibad"]["true_regimes"]  # (N,)
        # Simple: train logistic regression, measure accuracy
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression().fit(mu, regimes)
        acc = clf.score(mu, regimes)
        assert acc > 0.70, f"Latent regime classification accuracy too low: {acc:.2%}"

    def test_latent_regime_separation(self, eval_results):
        """Mean latent embedding should differ significantly across regimes."""
        mu = eval_results["varibad"]["latent_mu"]
        regimes = eval_results["varibad"]["true_regimes"]
        means = [mu[regimes == r].mean(axis=0) for r in [0, 1, 2]]
        # All pairwise L2 distances > 0.5
        for i in range(3):
            for j in range(i+1, 3):
                dist = np.linalg.norm(means[i] - means[j])
                assert dist > 0.5, f"Regimes {i},{j} not separated: L2={dist:.3f}"


class TestReproducibility:
    def test_variance_across_seeds(self, eval_results):
        """Reward std across 5 seeds should be < 30% of mean (results are stable)."""
        for agent in ["ppo", "rl2", "varibad", "varibad_hyper"]:
            rewards = eval_results[agent]["rewards"].mean(axis=1)  # (N_SEEDS,)
            cv = rewards.std() / (abs(rewards.mean()) + 1e-8)  # coefficient of variation
            assert cv < 0.30, (
                f"{agent} has high variance across seeds: cv={cv:.2%}"
            )
```

## Scripts

### `scripts/compare_agents.py`

```
python scripts/compare_agents.py --checkpoint_dir checkpoints/ --n_eval 200
```

Generates: `plots/comparison.png` (6-panel figure), `plots/learning_curves.png`, results table to stdout.

### `scripts/eval_agent.py`

```
python scripts/eval_agent.py --checkpoint checkpoints/varibad_42.eqx --n_episodes 100
```

Generates per-agent diagnostic plots.

### `scripts/ablation_persistence.py`

```
python scripts/ablation_persistence.py --checkpoint_dir checkpoints/ --n_eval 100
```

Generates: `plots/ablation_persistence.png` (VariBAD advantage vs regime duration).

### `scripts/train_all_seeds.py`

```
python scripts/train_all_seeds.py --n_iterations 1000 --n_envs 64
```

Trains PPO, RL², VariBAD, VariBAD+HN × 5 seeds. Saves learning curve data alongside checkpoints for plotting.

## Additional Dependencies (Phase 10 only)

```toml
dependencies = [..., "scipy", "scikit-learn"]
```

## Success Criteria

All tests pass: `pytest tests/test_phase10.py -v`. All plots generated. Specifically:

- [ ] VariBAD significantly outperforms PPO (p < 0.05 Wilcoxon)
- [ ] All learned agents below oracle ceiling
- [ ] VariBAD latent regime classification > 70% accuracy
- [ ] Cross-seed variance < 30% coefficient of variation
- [ ] Adaptation speed plot shows VariBAD recovers faster after switches
- [ ] Learning curves show convergence for all agents
- [ ] Persistence ablation shows expected U-shaped advantage curve
- [ ] Results table with p-values and Cohen's d printed

---

# Running All Tests

To verify the full project at any point:

```bash
# Run all tests for completed phases
pytest tests/ -v

# Run a specific phase
pytest tests/test_phase3.py -v

# Run with output visible (for performance numbers)
pytest tests/test_phase5.py -v -s

# Run everything except slow tests
pytest tests/ -v -k "not phase4 and not phase10"
```