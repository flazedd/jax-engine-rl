"""Linear-probe + posterior-quality metrics for RQ2.

Given a method's per-step belief representation (RL² hidden state, VariBAD μ,
Belief-PPO posterior, ...) plus the true regime and the analytical HMM
posterior at every step, this module:

1. Trains a linear (multinomial logistic) probe on (representation → regime)
   to summarize the regime-classification accuracy of the representation.
2. Reports posterior-quality metrics — symmetric KL and squared error —
   between the probe-projected posterior and the analytical posterior.

This is post-hoc analysis: probes train on collected rollout features, not
during agent training. Data collection happens in the orchestrator.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import optimize as spo


def _multinomial_logreg(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    l2: float = 1e-3,
    max_iter: int = 200,
) -> np.ndarray:
    """Fit a small multinomial logistic regression by L-BFGS-B on the full
    cross-entropy loss with L2 regularization. Returns a [D+1, K] weight
    matrix where the last row is the bias.

    No sklearn dependency — keeps the eval module lightweight.
    """
    n, d = X.shape
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)  # [N, D+1]
    Y = np.eye(n_classes, dtype=np.float64)[y]         # [N, K]

    def loss_and_grad(w_flat: np.ndarray) -> tuple[float, np.ndarray]:
        W = w_flat.reshape(d + 1, n_classes)
        logits = Xb @ W                                 # [N, K]
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        Z = exp.sum(axis=1, keepdims=True)
        log_probs = logits - np.log(Z)
        nll = -(Y * log_probs).sum() / n
        # L2 on slope rows only (skip the bias).
        nll += 0.5 * l2 * (W[:-1] ** 2).sum()
        probs = exp / Z
        grad = Xb.T @ (probs - Y) / n                   # [D+1, K]
        grad[:-1] += l2 * W[:-1]
        return float(nll), grad.ravel()

    w0 = np.zeros((d + 1) * n_classes, dtype=np.float64)
    res = spo.minimize(
        loss_and_grad, w0, jac=True, method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    return res.x.reshape(d + 1, n_classes)


def train_linear_probe(
    features: np.ndarray,
    regimes: np.ndarray,
    n_classes: int,
    l2: float = 1e-3,
    train_frac: float = 0.7,
    seed: int = 0,
) -> dict[str, object]:
    """Train a linear probe on (features → regime).

    Returns:
      {"W": [D+1, K], "predict": fn, "test_idx": ..., "train_idx": ...}.
    `predict(X) -> [N, K]` returns the simplex-projected posterior.
    """
    n = features.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(np.floor(train_frac * n))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = features[train_idx], regimes[train_idx]
    W = _multinomial_logreg(X_train, y_train, n_classes=n_classes, l2=l2)

    def predict(X: np.ndarray) -> np.ndarray:
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        logits = Xb @ W
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    return {"W": W, "predict": predict, "train_idx": train_idx, "test_idx": test_idx}


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, 1.0))


def symmetric_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row symmetric KL between two simplex-valued distributions."""
    return 0.5 * (
        (p * (_safe_log(p) - _safe_log(q))).sum(axis=-1)
        + (q * (_safe_log(q) - _safe_log(p))).sum(axis=-1)
    )


def evaluate_probe(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    features: np.ndarray,
    regimes: np.ndarray,
    analytical_post: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate a trained probe on held-out (or full) rollout features.

    Returns:
      classification_accuracy: argmax(predicted) == regime, mean.
      posterior_mse: ||predicted - analytical||² mean (NaN if analytical not given).
      symmetric_kl: mean symmetric KL (NaN if analytical not given).
    """
    pred = predict_fn(features)
    acc = float((pred.argmax(axis=1) == regimes).mean())
    out: dict[str, float] = {"classification_accuracy": acc}
    if analytical_post is None:
        out["posterior_mse"] = float("nan")
        out["symmetric_kl"] = float("nan")
        return out
    out["posterior_mse"] = float(((pred - analytical_post) ** 2).sum(axis=-1).mean())
    out["symmetric_kl"] = float(symmetric_kl(pred, analytical_post).mean())
    return out


def evaluate_method(
    features: np.ndarray | None,
    regimes: np.ndarray,
    n_classes: int,
    analytical_post: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """End-to-end: train probe on (features, regimes), evaluate on test split.

    For methods without a belief representation (`features is None`), the
    probe is undefined and metrics are NaN — these methods are still
    runnable but do not contribute to `posterior_quality_discrimination`.
    """
    if features is None:
        return {
            "classification_accuracy": float("nan"),
            "posterior_mse": float("nan"),
            "symmetric_kl": float("nan"),
            "n_features": 0,
        }
    probe = train_linear_probe(
        features, regimes, n_classes=n_classes, seed=seed
    )
    test_idx = probe["test_idx"]
    metrics = evaluate_probe(
        probe["predict"],
        features[test_idx],
        regimes[test_idx],
        None if analytical_post is None else analytical_post[test_idx],
    )
    metrics["n_features"] = int(features.shape[0])
    return metrics
