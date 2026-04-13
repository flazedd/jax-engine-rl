"""Shared components for all agents."""
import json

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np


class MLP(eqx.Module):
    """Generic feedforward network with ReLU hidden layers."""
    layers: list

    def __init__(self, sizes, *, key):
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [eqx.nn.Linear(a, b, key=k)
                       for a, b, k in zip(sizes[:-1], sizes[1:], keys)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, bootstrap_value, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation via reverse scan.

    Args:
        rewards: (T,) per-step rewards
        values: (T,) value estimates
        dones: (T,) terminal flags (1.0 = terminal)
        bootstrap_value: scalar value estimate at step T
        gamma: discount factor
        lam: GAE lambda
    Returns:
        advantages: (T,)
        returns: (T,) = advantages + values
    """
    T = rewards.shape[0]

    def scan_fn(gae, t):
        idx = T - 1 - t
        next_val = jnp.where(idx < T - 1, values[idx + 1], bootstrap_value)
        delta = rewards[idx] + gamma * next_val * (1 - dones[idx]) - values[idx]
        gae = delta + gamma * lam * (1 - dones[idx]) * gae
        return gae, gae

    _, adv_rev = jax.lax.scan(scan_fn, jnp.float32(0.0), jnp.arange(T))
    adv = adv_rev[::-1]
    return adv, adv + values


batch_gae = jax.vmap(compute_gae, in_axes=(1, 1, 1, 0, None, None),
                     out_axes=(1, 1))


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.bool_)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
