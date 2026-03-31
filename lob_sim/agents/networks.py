"""Neural network primitives using equinox."""
import jax
import jax.numpy as jnp
import equinox as eqx


class MLP(eqx.Module):
    layers: tuple

    def __init__(self, in_size: int, hidden_sizes: list[int], out_size: int, *, key):
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        sizes = [in_size] + hidden_sizes + [out_size]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i]))
        self.layers = tuple(layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        return x


class GRUCell(eqx.Module):
    """Single GRU step. Takes (input, hidden) -> new_hidden."""
    cell: eqx.nn.GRUCell

    def __init__(self, input_size: int, hidden_size: int, *, key):
        self.cell = eqx.nn.GRUCell(input_size, hidden_size, key=key)

    def __call__(self, x: jnp.ndarray, hidden: jnp.ndarray) -> jnp.ndarray:
        return self.cell(x, hidden)
