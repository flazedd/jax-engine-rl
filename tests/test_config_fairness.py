"""Fairness invariants for every evaluated comparison.

Checks:
  (1) Each evaluated family passes the audit: matched inputs, matched
      optimiser settings, matched training budget, matched capacity, and an
      encoder shared across each conditioning pair.
  (2) The audit is not vacuous: the pre-matched configs, where the references
      are memoryless on o_t and run a different optimiser setting, are
      reported as violations.
"""
from __future__ import annotations

from scripts.config_fairness_audit import FAMILIES, audit_family

TOL = 0.025

# The configs used before inputs and optimiser settings were matched. Retained
# only as the negative control for check (2).
LEGACY_SPEC = {
    "budget": 5000,
    "base_obs": 11,
    "methods": [
        ("regime_agnostic", "m3_regime_agnostic.yaml", "reference"),
        ("belief_ppo", "m3_belief.yaml", "regime"),
        ("oracle_ppo", "m3_oracle.yaml", "regime"),
        ("rl2_concat", "m5r_locked/rl2_concat.yaml", "meta"),
        ("rl2_hypernet", "m5r_locked/rl2_hypernet.yaml", "meta"),
        ("varibad_concat", "m5r_locked/varibad_concat.yaml", "meta"),
        ("varibad_hypernet", "m5r_locked/varibad_hypernet.yaml", "meta"),
    ],
}


def test_evaluated_families_are_fair():
    for name, spec in FAMILIES.items():
        report = audit_family(name, spec, TOL)
        assert not report["violations"], (
            f"{name} is not a fair comparison:\n  "
            + "\n  ".join(report["violations"])
        )


def test_pair_encoders_are_identical():
    """The conditioning ablation must vary conditioning only."""
    for name, spec in FAMILIES.items():
        report = audit_family(name, spec, TOL)
        rows = report["rows"]
        for a, b in (("rl2_concat", "rl2_hypernet"),
                     ("varibad_concat", "varibad_hypernet")):
            assert rows[a]["encoder_spec"] == rows[b]["encoder_spec"], (
                f"{name}: {a} and {b} do not share an encoder"
            )
            assert rows[a]["obs_size"] == rows[b]["obs_size"], (
                f"{name}: {a} and {b} do not read the same inputs"
            )


def test_audit_detects_the_pre_matched_configs():
    report = audit_family("legacy", LEGACY_SPEC, TOL)
    violations = "\n".join(report["violations"])
    assert report["violations"], "audit passed a comparison known to be unfair"
    assert "obs_size" in violations, "input mismatch not detected"
    assert "optimiser.epochs" in violations, "optimiser mismatch not detected"
