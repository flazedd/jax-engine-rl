"""Hypernet integration module — composable axis for RL² and VariBAD.

Replaces the concat path: instead of concatenating belief into the policy
input, the belief generates the *weights* of a small target policy. The
target maps the agent's policy input through the belief-generated weights
to produce action logits.

Spec lives at docs/implementation.md → "agents/modules/hypernet.py".
"""
from __future__ import annotations

import chex
import flax.linen as nn
import jax.numpy as jnp


class Hypernet(nn.Module):
    """Maps a belief vector to the weights of a 2-layer target MLP.

    Target architecture: input → tanh-Dense(target_hidden) → Dense(target_output_dim).
    Only the *weights* are belief-conditional; the architecture is fixed.

    The target's input is whatever the agent normally feeds its policy
    (augmented obs for RL², base obs for VariBAD). `target_obs_dim` is that
    input's dimension.
    """

    target_obs_dim: int
    target_hidden: int
    target_output_dim: int  # action_dim
    hypernet_hidden: int
    init_scale: float = 0.01
    # Number of hidden layers in the generated target. Set equal to the concat
    # arm's policy depth so the observation-to-action map has the same shape in
    # both conditioning architectures; that path is the policy function itself,
    # and matching it is what makes the two arms comparable over observations.
    target_hidden_layers: int = 1

    def _target_shapes(self) -> list[tuple[int, int]]:
        """(in_dim, out_dim) of each Dense layer in the generated target."""
        dims = (
            [self.target_obs_dim]
            + [self.target_hidden] * self.target_hidden_layers
            + [self.target_output_dim]
        )
        return list(zip(dims[:-1], dims[1:]))

    def target_param_count(self) -> int:
        """Total scalar weights needed to parameterize the target network."""
        return sum(d_in * d_out + d_out for d_in, d_out in self._target_shapes())

    @nn.compact
    def __call__(self, belief: chex.Array) -> chex.Array:
        """Belief → flat weight vector for the target."""
        h = nn.Dense(
            self.hypernet_hidden,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
        )(belief)
        h = nn.tanh(h)
        # Final-layer init scale is exposed because it is the integration-care
        # analog of LayerNorm on the concat side: 0.0 = Beck et al. zero-init
        # (all beliefs map to a single shared target at start); 0.01 = current
        # default; larger = more belief-driven variance at init.
        return nn.Dense(
            self.target_param_count(),
            kernel_init=nn.initializers.orthogonal(self.init_scale),
        )(h)

    def apply_target(
        self, flat_weights: chex.Array, obs: chex.Array
    ) -> chex.Array:
        """Unpack flat weights into target tensors and run the forward pass.

        flat_weights: [target_param_count]. obs: [target_obs_dim].
        Returns logits: [target_output_dim].

        Hidden layers use tanh; the output layer is linear, matching the concat
        arm's policy MLP.
        """
        shapes = self._target_shapes()
        i = 0
        h = obs
        for layer, (d_in, d_out) in enumerate(shapes):
            W = flat_weights[i : i + d_in * d_out].reshape(d_in, d_out)
            i += d_in * d_out
            b = flat_weights[i : i + d_out]
            i += d_out
            h = h @ W + b
            if layer < len(shapes) - 1:
                h = jnp.tanh(h)
        return h
