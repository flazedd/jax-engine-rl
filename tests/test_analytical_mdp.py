"""Tests for the analytical market making MDP with Bellman equation solutions."""
import numpy as np
import pytest

from lob_sim.analytical_mdp import (
    MDPConfig,
    build_mdp_tables,
    compute_fill_probs,
    solve_full_info,
    solve_pomdp_belief,
    value_of_info,
    _build_belief_grid,
    _nearest_belief,
    _bayesian_update,
    ACTION_TABLE_ANALYTICAL,
    N_ACTIONS_ANALYTICAL,
    N_REGIMES,
)


@pytest.fixture
def cfg():
    return MDPConfig()


@pytest.fixture
def tables(cfg):
    return build_mdp_tables(cfg)


@pytest.fixture
def sol(tables):
    return solve_full_info(tables)


# ── Fill model ───────────────────────────────────────────────

class TestFillModel:
    def test_shape(self, cfg):
        fills = compute_fill_probs(cfg)
        assert fills.bid.shape == (N_REGIMES, N_ACTIONS_ANALYTICAL)
        assert fills.ask.shape == (N_REGIMES, N_ACTIONS_ANALYTICAL)

    def test_valid_probabilities(self, cfg):
        fills = compute_fill_probs(cfg)
        assert np.all(fills.bid >= 0) and np.all(fills.bid <= 1)
        assert np.all(fills.ask >= 0) and np.all(fills.ask <= 1)

    def test_tighter_offset_higher_fill(self, cfg):
        """Tighter quotes (smaller offset) should fill more often."""
        fills = compute_fill_probs(cfg)
        # a1=(1,9): bid offset=1 (tight), a2=(9,1): bid offset=9 (wide)
        for r in range(N_REGIMES):
            assert fills.bid[r, 1] > fills.bid[r, 2], \
                f"Regime {r}: tight bid should fill more"
            assert fills.ask[r, 2] > fills.ask[r, 1], \
                f"Regime {r}: tight ask should fill more"

    def test_bull_more_bid_fills(self, cfg):
        """Bull regime has more market sells → more bid fills."""
        fills = compute_fill_probs(cfg)
        for a in range(N_ACTIONS_ANALYTICAL):
            assert fills.bid[1, a] > fills.bid[2, a], \
                f"Action {a}: Bull should have higher bid fill rate than Bear"

    def test_bear_more_ask_fills(self, cfg):
        """Bear regime has more market buys → more ask fills."""
        fills = compute_fill_probs(cfg)
        for a in range(N_ACTIONS_ANALYTICAL):
            assert fills.ask[2, a] > fills.ask[1, a], \
                f"Action {a}: Bear should have higher ask fill rate than Bull"

    def test_noise_symmetric(self, cfg):
        """Noise regime should have equal bid and ask fill rates."""
        fills = compute_fill_probs(cfg)
        # a0=(2,2): symmetric offsets → identical fills
        np.testing.assert_allclose(fills.bid[0, 0], fills.ask[0, 0])
        # a1 bid = a2 ask (by symmetry of offsets and equal arrival rates)
        np.testing.assert_allclose(fills.bid[0, 1], fills.ask[0, 2])


# ── MDP tables ───────────────────────────────────────────────

class TestMDPTables:
    def test_reward_shape(self, tables, cfg):
        n_inv = 2 * cfg.max_inv + 1
        assert tables.reward.shape == (N_REGIMES, n_inv, N_ACTIONS_ANALYTICAL)

    def test_transition_shape(self, tables, cfg):
        n_inv = 2 * cfg.max_inv + 1
        assert tables.trans_inv.shape == (
            N_REGIMES, n_inv, N_ACTIONS_ANALYTICAL, n_inv)

    def test_transition_sums_to_one(self, tables):
        sums = tables.trans_inv.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-12)

    def test_regime_transition_sums(self, tables):
        sums = tables.trans_regime.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-12)

    def test_no_nan(self, tables):
        assert not np.any(np.isnan(tables.reward))
        assert not np.any(np.isnan(tables.trans_inv))

    def test_reward_penalty_at_extremes(self, tables, cfg):
        """In Noise (zero drift), reward at extreme inventory < at zero."""
        mi = cfg.max_inv
        r = 0  # Noise only — drift regimes can have higher R at extremes
        for a in range(N_ACTIONS_ANALYTICAL):
            assert tables.reward[r, 0, a] < tables.reward[r, mi, a], \
                f"R(Noise, q=-{mi}, a={a}) should be less than R(Noise, q=0, a={a})"
            assert tables.reward[r, 2*mi, a] < tables.reward[r, mi, a], \
                f"R(Noise, q=+{mi}, a={a}) should be less than R(Noise, q=0, a={a})"

    def test_noise_reward_symmetric(self, tables, cfg):
        """In Noise regime, R(q, a0) should equal R(-q, a0)."""
        mi = cfg.max_inv
        R = tables.reward[0]  # Noise
        for q in range(1, mi + 1):
            np.testing.assert_allclose(
                R[mi + q, 0], R[mi - q, 0], atol=1e-12,
                err_msg=f"Noise R should be symmetric at q=±{q}")


# ── Full-info value iteration ────────────────────────────────

class TestFullInfoVI:
    def test_converges(self, sol):
        assert sol.residuals[-1] < 1e-8

    def test_policy_shape(self, sol, cfg):
        n_inv = 2 * cfg.max_inv + 1
        assert sol.policy.shape == (N_REGIMES, n_inv)
        assert sol.values.shape == (N_REGIMES, n_inv)
        assert sol.Q.shape == (N_REGIMES, n_inv, N_ACTIONS_ANALYTICAL)

    def test_noise_symmetric_policy(self, sol, cfg):
        """Noise policy should be symmetric: π(q) mirrors π(-q)."""
        mi = cfg.max_inv
        p = sol.policy[0]  # Noise
        for q in range(1, mi + 1):
            a_pos = p[mi + q]
            a_neg = p[mi - q]
            # a1 (go long) at -q should mirror a2 (go short) at +q
            if a_neg == 1:
                assert a_pos == 2, f"Noise policy asymmetric at q=±{q}"
            elif a_neg == 2:
                assert a_pos == 1, f"Noise policy asymmetric at q=±{q}"
            else:
                assert a_pos == a_neg, f"Noise policy asymmetric at q=±{q}"

    def test_regimes_differ_at_inv_zero(self, sol, cfg):
        """Each regime should have a different optimal action at inv=0."""
        mi = cfg.max_inv
        actions = set()
        for r in range(N_REGIMES):
            actions.add(int(sol.policy[r, mi]))
        assert len(actions) == N_REGIMES, \
            "Optimal actions at inv=0 should differ across all regimes"

    def test_bull_goes_long(self, sol, cfg):
        """Bull at inv=0 should play action 1 (tight bid = go long)."""
        mi = cfg.max_inv
        assert sol.policy[1, mi] == 1, \
            f"Bull should play a1=(1,3), got a{sol.policy[1, mi]}"

    def test_bear_goes_short(self, sol, cfg):
        """Bear at inv=0 should play action 2 (tight ask = go short)."""
        mi = cfg.max_inv
        assert sol.policy[2, mi] == 2, \
            f"Bear should play a2=(3,1), got a{sol.policy[2, mi]}"

    def test_noise_plays_symmetric(self, sol, cfg):
        """Noise at inv=0 should play action 0 (symmetric)."""
        mi = cfg.max_inv
        assert sol.policy[0, mi] == 0, \
            f"Noise should play a0=(2,2), got a{sol.policy[0, mi]}"

    def test_noise_value_peaks_near_zero(self, sol, cfg):
        """Noise value function should peak near inv=0 and decay at extremes."""
        mi = cfg.max_inv
        V = sol.values[0]
        # Peak is at inv=0
        assert np.argmax(V) == mi, "Noise V should peak at inv=0"
        # Extremes are lower
        assert V[0] < V[mi] and V[-1] < V[mi]

    def test_bull_bear_value_symmetry(self, sol, cfg):
        """V_bull(q) should equal V_bear(-q) by symmetry."""
        mi = cfg.max_inv
        for q in range(-mi, mi + 1):
            np.testing.assert_allclose(
                sol.values[1, q + mi], sol.values[2, -q + mi],
                atol=1e-8,
                err_msg=f"Bull V(q={q}) should equal Bear V(q={-q})")

    def test_fast_solve_time(self, tables):
        """Full-info VI should solve in <50ms."""
        import time
        t0 = time.time()
        solve_full_info(tables)
        assert time.time() - t0 < 0.05


# ── POMDP ────────────────────────────────────────────────────

class TestPOMDP:
    @pytest.fixture
    def small_cfg(self):
        return MDPConfig(max_inv=5, n_belief_points=21)

    @pytest.fixture
    def small_tables(self, small_cfg):
        return build_mdp_tables(small_cfg)

    @pytest.fixture
    def small_sol(self, small_tables):
        return solve_full_info(small_tables)

    @pytest.fixture
    def pomdp_sol(self, small_tables):
        return solve_pomdp_belief(small_tables)

    def test_belief_grid_valid(self):
        grid = _build_belief_grid(11)
        np.testing.assert_allclose(grid.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(grid >= 0)

    def test_bayesian_update_normalizes(self, small_tables):
        fills = small_tables.fill_probs
        T_hmm = small_tables.trans_regime
        b = np.ones(N_REGIMES) / N_REGIMES
        for a in range(N_ACTIONS_ANALYTICAL):
            for ob in [0, 1]:
                for oa in [0, 1]:
                    b_new = _bayesian_update(b, ob, oa, a, fills, T_hmm)
                    np.testing.assert_allclose(b_new.sum(), 1.0, atol=1e-12)

    def test_pomdp_converges(self, pomdp_sol):
        assert pomdp_sol.residuals[-1] < 1e-6

    def test_positive_value_of_info(self, small_tables, small_sol, pomdp_sol):
        """Knowing the regime should always be weakly better."""
        voi = value_of_info(small_tables, small_sol, pomdp_sol)
        assert voi >= -1e-6, f"Value of info should be non-negative, got {voi}"

    def test_pomdp_value_below_full_info(self, small_tables, small_sol, pomdp_sol):
        """POMDP value at uniform belief should be ≤ average full-info value."""
        mi = small_tables.config.max_inv
        grid = _build_belief_grid(small_tables.config.n_belief_points)
        bi = _nearest_belief(np.ones(3) / 3, grid)
        v_pomdp = pomdp_sol.values[bi, mi]
        v_full_avg = np.mean(small_sol.values[:, mi])
        assert v_pomdp <= v_full_avg + 1e-6


# ── Cross-regime penalty ─────────────────────────────────────

class TestCrossRegimePenalty:
    def test_wrong_policy_worse(self, tables, cfg):
        """Playing a regime's policy in the wrong regime should give lower Q."""
        mi = cfg.max_inv
        sol = solve_full_info(tables)
        Q = sol.Q

        # Bull policy in Bear should be worse than Bear's own policy
        a_bull = sol.policy[1, mi]  # Bull's action at inv=0
        a_bear = sol.policy[2, mi]  # Bear's action at inv=0
        assert Q[2, mi, a_bear] > Q[2, mi, a_bull], \
            "Bear's own policy should beat Bull's policy in Bear regime"
        assert Q[1, mi, a_bull] > Q[1, mi, a_bear], \
            "Bull's own policy should beat Bear's policy in Bull regime"
