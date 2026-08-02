"""RL² (Duan et al., 2016) — recurrent meta-RL with PPO inner loop.

The policy is a GRU-actor-critic that consumes the augmented observation
`[obs, prev_action, prev_reward, prev_done]` (provided by `RL2ObsEnv`) plus
its hidden state. The hidden state is the agent's implicit "belief" about
the latent task; it resets between episodes (one episode = one task).

The PPO update replays the recurrent forward pass over the full rollout so
gradients can flow through the GRU. Minibatches shuffle along the env axis
only (T axis stays contiguous per env to preserve recurrence).
"""
from __future__ import annotations

from dataclasses import dataclass

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.modules.hypernet import Hypernet
from training.ppo_update import compute_gae


class GRUActorCritic(nn.Module):
    """RL² actor-critic: GRU encoder + policy and value heads.

    Two integration paths for the policy head, selected by `integration`:
    - "concat" (default): policy is Dense(hidden) → action_logits, operating
      on the GRU hidden state directly.
    - "hypernet": GRU hidden generates the weights of a small target MLP
      via Hypernet; target operates on the (augmented) obs to produce
      logits. Value head stays Dense(hidden) regardless.
    """

    n_actions: int
    obs_size: int
    hidden_dim: int = 64
    # When True, the policy and value heads read a stop-gradient copy of the GRU
    # belief, so their gradients do not shape the encoder. Combined with an
    # auxiliary regime-decode loss, the encoder is then trained purely for
    # decodability while a hypernet policy exploits it (single-stage decoupling).
    detach_belief_for_policy: bool = False
    integration: str = "concat"
    hypernet_target_hidden: int = 16
    hypernet_hidden: int = 64
    # Concat-side architectural-care knob: LayerNorm the GRU hidden (the
    # implicit "belief") before the policy head. No-op for the hypernet path.
    belief_layernorm: bool = False
    # Hypernet-side architectural-care knob: init scale on the hypernet output
    # layer. 0.0 = Beck et al. zero-init.
    hypernet_init_scale: float = 0.01
    # Optional policy trunk inserted between the GRU output and the policy
    # head (concat) or hypernet input (hypernet). Used to equalise total
    # parameter counts across the matched-compute factorial. The value head
    # always reads the GRU output directly, so the trunk only affects the
    # policy path.
    policy_trunk_layers: int = 0
    policy_trunk_hidden: int = 0
    # 3-simplex belief bottleneck: the policy conditions on a learned softmax
    # over regimes instead of the raw hidden state, giving concat a calibrated
    # low-dimensional belief like the analytical posterior. The value head keeps
    # reading the full hidden state.
    belief_simplex_head: bool = False
    n_regimes: int = 3
    # Matched policy inputs. The encoder always consumes the augmented
    # observation [o_t, a_{t-1}, r_{t-1}]; these two knobs control what the
    # *policy* sees, so that concat and hypernet differ only in how the belief
    # enters and not in what information reaches the action.
    #   policy_obs_dim > 0 restricts the policy-side observation to its first
    #     `policy_obs_dim` entries, i.e. the base observation o_t without the
    #     previous action and reward, as specified in the methodology.
    #   concat_policy_reads_obs gives the concat path o_t alongside the belief,
    #     matching what the hypernet target network already receives.
    # Both default to the legacy behaviour so existing checkpoints load.
    policy_obs_dim: int = 0
    concat_policy_reads_obs: bool = False

    def _policy_obs(self, obs: chex.Array) -> chex.Array:
        if self.policy_obs_dim <= 0:
            return obs
        return obs[..., : self.policy_obs_dim]

    @nn.compact
    def __call__(
        self, carry: chex.Array, obs: chex.Array
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
        )(obs)
        x = nn.tanh(x)
        new_carry, _ = nn.GRUCell(features=self.hidden_dim)(carry, x)

        # The heads optionally read a detached belief so their gradients do not
        # flow back into the encoder; the raw new_carry is still returned for the
        # recurrence and for the auxiliary decoder.
        belief = (jax.lax.stop_gradient(new_carry)
                  if self.detach_belief_for_policy else new_carry)

        if self.belief_simplex_head:
            simplex_logits = nn.Dense(
                self.n_regimes,
                kernel_init=nn.initializers.orthogonal(0.01),
                name="simplex_head",
            )(belief)
            policy_belief = jax.nn.softmax(simplex_logits)
        else:
            simplex_logits = jnp.zeros((self.n_regimes,), dtype=new_carry.dtype)
            policy_belief = belief

        policy_obs = self._policy_obs(obs)

        trunk_out = (
            jnp.concatenate([policy_obs, policy_belief], axis=-1)
            if (self.integration == "concat" and self.concat_policy_reads_obs)
            else policy_belief
        )
        for _ in range(self.policy_trunk_layers):
            trunk_out = nn.Dense(
                self.policy_trunk_hidden,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
            )(trunk_out)
            trunk_out = nn.tanh(trunk_out)

        if self.integration == "concat":
            policy_in = nn.LayerNorm()(trunk_out) if self.belief_layernorm else trunk_out
            logits = nn.Dense(
                self.n_actions,
                kernel_init=nn.initializers.orthogonal(0.01),
            )(policy_in)
        elif self.integration == "hypernet":
            hn = Hypernet(
                target_obs_dim=(self.policy_obs_dim or self.obs_size),
                target_hidden=self.hypernet_target_hidden,
                target_output_dim=self.n_actions,
                hypernet_hidden=self.hypernet_hidden,
                init_scale=self.hypernet_init_scale,
            )
            flat_weights = hn(trunk_out)
            logits = hn.apply_target(flat_weights, policy_obs)
        else:
            raise ValueError(f"unknown integration: {self.integration!r}")

        # Under matched policy inputs the actor reads [o_t, h_t]; the critic
        # reads the same, otherwise it must recover the inventory from the
        # recurrent state that the actor is handed directly, and its noisier
        # advantages would differ between variants for reasons unrelated to the
        # conditioning architecture.
        value_in = (
            jnp.concatenate([policy_obs, belief], axis=-1)
            if self.concat_policy_reads_obs else belief
        )
        value = nn.Dense(
            1, kernel_init=nn.initializers.orthogonal(1.0)
        )(value_in).squeeze(-1)
        return new_carry, logits, value, simplex_logits


@dataclass(frozen=True)
class RL2Agent:
    obs_size: int
    n_actions: int
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95
    epochs: int = 4
    minibatch_envs: int = 32
    # Policy objective: "ppo" (clipped surrogate, default) or "a2c" (vanilla
    # advantage actor-critic, no importance ratio and no clipping). Used to
    # test whether the integration finding is specific to PPO; A2C should be
    # run with epochs=1 to stay on-policy.
    policy_objective: str = "ppo"
    # Integration mechanism: "concat" = Dense(hidden)→logits;
    # "hypernet" = GRU hidden generates target-MLP weights, target maps obs→logits.
    integration: str = "concat"
    hypernet_target_hidden: int = 16
    hypernet_hidden: int = 64
    # Architectural-care knobs (defaults preserve previous behaviour).
    belief_layernorm: bool = False
    hypernet_init_scale: float = 0.01
    # Matched-compute knobs: optional policy trunk between GRU output and
    # action / hypernet head. Defaults of 0 disable the trunk.
    policy_trunk_layers: int = 0
    policy_trunk_hidden: int = 0

    requires_regime_label: bool = False
    requires_analytical_posterior: bool = False
    is_recurrent: bool = True
    produces_belief_for_eval: bool = False
    # Trajectory key carrying this method's belief vector. Used by the
    # posterior-quality probe (evaluation/posterior_probe.py) to identify
    # the agent's internal belief representation.
    belief_key: str = "carry_in"

    # Single-stage "best of both" knob: an auxiliary regime-decoding loss on
    # the GRU belief, added to the PPO loss with this coefficient (0 = off, no
    # aux params, behaviour identical to before). Pressures the encoder to keep
    # the regime linearly decodable while the policy uses any integration
    # mechanism (e.g. hypernet), aiming for concat-quality belief + hypernet
    # usage in one training run.
    aux_decode_coef: float = 0.0
    n_regimes: int = 3
    # Detach the belief into the policy/value heads (single-stage decoupling);
    # only meaningful together with `aux_decode_coef > 0`.
    detach_belief_for_policy: bool = False
    # 3-simplex belief bottleneck (see GRUActorCritic). When on, the policy
    # conditions on a learned softmax over regimes; with aux_decode_coef > 0 that
    # softmax is supervised toward the true regime. Off by default.
    belief_simplex_head: bool = False
    # Matched policy inputs; see GRUActorCritic for the semantics.
    policy_obs_dim: int = 0
    concat_policy_reads_obs: bool = False

    # ---- model factory --------------------------------------------------

    def _model(self) -> GRUActorCritic:
        return GRUActorCritic(
            n_actions=self.n_actions,
            obs_size=self.obs_size,
            hidden_dim=self.hidden_dim,
            integration=self.integration,
            hypernet_target_hidden=self.hypernet_target_hidden,
            hypernet_hidden=self.hypernet_hidden,
            belief_layernorm=self.belief_layernorm,
            hypernet_init_scale=self.hypernet_init_scale,
            policy_trunk_layers=self.policy_trunk_layers,
            policy_trunk_hidden=self.policy_trunk_hidden,
            detach_belief_for_policy=self.detach_belief_for_policy,
            belief_simplex_head=self.belief_simplex_head,
            n_regimes=self.n_regimes,
            policy_obs_dim=self.policy_obs_dim,
            concat_policy_reads_obs=self.concat_policy_reads_obs,
        )

    def _optimizer(self) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.learning_rate),
        )

    # ---- Agent API ------------------------------------------------------

    def init(self, key: chex.PRNGKey) -> chex.ArrayTree:
        model = self._model()
        k_model, k_aux = jax.random.split(key)
        dummy_carry = jnp.zeros((self.hidden_dim,), dtype=jnp.float32)
        dummy_obs = jnp.zeros((self.obs_size,), dtype=jnp.float32)
        params = model.init(k_model, dummy_carry, dummy_obs)
        if self.aux_decode_coef > 0.0 and not self.belief_simplex_head:
            # Linear regime decoder over the GRU belief; trained jointly, lives
            # as a sibling collection so the model (which only reads "params")
            # is untouched while the optimizer still updates it.
            params = dict(params)
            params["aux"] = {
                "W": nn.initializers.orthogonal(1.0)(
                    k_aux, (self.hidden_dim, self.n_regimes)),
                "b": jnp.zeros((self.n_regimes,), dtype=jnp.float32),
            }
        opt_state = self._optimizer().init(params)
        return {
            "params": params,
            "opt_state": opt_state,
            "update_key": jax.random.PRNGKey(0),
        }

    def init_carry(self, n_parallel: int) -> chex.Array:
        return jnp.zeros((n_parallel, self.hidden_dim), dtype=jnp.float32)

    def act(
        self,
        state: chex.ArrayTree,
        carry: chex.Array,
        obs: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.Array, dict[str, chex.Array], chex.Array]:
        new_carry, logits, value, simplex_logits = self._model().apply(
            {"params": state["params"]["params"]}, carry, obs)
        action = jax.random.categorical(key, logits)
        log_prob = jax.nn.log_softmax(logits)[action]
        extras = {"log_prob": log_prob, "value": value}
        if self.belief_simplex_head:
            extras["belief_simplex"] = jax.nn.softmax(simplex_logits)
        return action, extras, new_carry

    # ---- update ---------------------------------------------------------

    def _replay_forward(self, params, init_carry, obs_seq, done_seq):
        """Replay the recurrent forward pass for one env's trajectory.

        init_carry: [hidden_dim]. obs_seq: [T, obs_size]. done_seq: [T] bool.
        Returns logits[T, n_actions], values[T], carries[T, hidden_dim] (the
        per-step post-update belief the policy reads; used by the aux decoder).
        """
        model = self._model()
        mvars = {"params": params["params"]}

        def step(carry, inputs):
            obs_t, done_t = inputs
            new_carry, logits_t, value_t, simplex_t = model.apply(mvars, carry, obs_t)
            # If this step ended an episode, reset carry for next step.
            zeros = jnp.zeros_like(new_carry)
            mask = done_t.astype(new_carry.dtype)
            next_carry = mask * zeros + (1.0 - mask) * new_carry
            return next_carry, (logits_t, value_t, new_carry, simplex_t)

        _, (logits, values, carries, simplex_logits) = jax.lax.scan(
            step, init_carry, (obs_seq, done_seq))
        return logits, values, carries, simplex_logits

    def _bootstrap_value(self, params, final_obs, final_carry):
        """Compute V(final_obs | final_carry) per env for GAE bootstrap."""
        mvars = {"params": params["params"]}

        def f(carry_i, obs_i):
            _, _, v, _ = self._model().apply(mvars, carry_i, obs_i)
            return v
        return jax.vmap(f)(final_carry, final_obs)

    def update(
        self,
        state: chex.ArrayTree,
        trajectory: chex.ArrayTree,
        final_obs: chex.Array,
        init_carry: chex.Array,
        final_carry: chex.Array,
    ) -> tuple[chex.ArrayTree, dict[str, chex.Array]]:
        params = state["params"]
        opt_state = state["opt_state"]

        last_value = self._bootstrap_value(params, final_obs, final_carry)
        advantages, returns = compute_gae(
            trajectory["reward"],
            trajectory["value"],
            trajectory["done"],
            last_value,
            gamma=self.gamma,
            lam=self.lam,
        )

        # Per-env data: shapes [T, N, ...]; init_carry [N, hidden_dim].
        T, N = trajectory["reward"].shape
        # Effective minibatch size: cap to N when N < requested.
        mb_size = min(self.minibatch_envs, N)

        def loss_fn(params, batch):
            # batch leaves: obs [T, n_envs, obs_size], action [T, n_envs],
            # log_prob [T, n_envs], advantage [T, n_envs], return [T, n_envs],
            # done [T, n_envs], init_carry [n_envs, hidden_dim].
            replay = jax.vmap(
                lambda c, o, d: self._replay_forward(params, c, o, d),
                in_axes=(0, 1, 1), out_axes=1,
            )
            logits, values, carries, simplex_logits = replay(
                batch["init_carry"], batch["obs"], batch["done"])
            log_probs_all = jax.nn.log_softmax(logits)
            lp_new = jnp.take_along_axis(
                log_probs_all, batch["action"][..., None], axis=-1
            ).squeeze(-1)

            ratio = jnp.exp(lp_new - batch["log_prob"])
            adv = batch["advantage"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            if self.policy_objective == "a2c":
                # Vanilla advantage actor-critic: plain policy-gradient, no
                # importance ratio and no clipping (run with epochs=1 to keep
                # the update on-policy).
                policy_loss = -(lp_new * adv).mean()
            else:
                unclipped = ratio * adv
                clipped = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -jnp.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * jnp.mean((values - batch["return"]) ** 2)
            probs = jnp.exp(log_probs_all)
            entropy = -jnp.sum(probs * log_probs_all, axis=-1).mean()
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            aux_loss = jnp.array(0.0)
            aux_acc = jnp.array(0.0)
            if self.aux_decode_coef > 0.0:
                # Regime cross-entropy. With a simplex head it supervises the
                # softmax the policy itself conditions on; otherwise it trains a
                # separate linear decoder over the GRU belief, pressuring the
                # encoder to keep the regime decodable while the policy uses its
                # own integration.
                if self.belief_simplex_head:
                    aux_logits = simplex_logits
                else:
                    aux_logits = (
                        jnp.einsum("tnh,hr->tnr", carries, params["aux"]["W"])
                        + params["aux"]["b"]
                    )
                aux_loss = optax.softmax_cross_entropy_with_integer_labels(
                    aux_logits, batch["regime"]
                ).mean()
                aux_acc = (aux_logits.argmax(-1) == batch["regime"]).mean()
                loss = loss + self.aux_decode_coef * aux_loss

            approx_kl = (batch["log_prob"] - lp_new).mean()
            clipped_frac = (jnp.abs(ratio - 1.0) > self.clip_eps).astype(jnp.float32).mean()
            metrics = {
                "ppo/policy_loss": policy_loss,
                "ppo/value_loss": value_loss,
                "ppo/entropy": entropy,
                "aux/decode_loss": aux_loss,
                "aux/decode_acc": aux_acc,
                "ppo/approx_kl": approx_kl,
                "ppo/clipped_frac": clipped_frac,
                "ppo/total_loss": loss,
            }
            return loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        optimizer = self._optimizer()

        # Shuffle along env axis only; T-axis stays contiguous to preserve
        # the GRU's temporal structure inside each minibatch.
        n_minibatches = max(1, N // mb_size)
        n_envs_used = n_minibatches * mb_size

        update_key, next_key = jax.random.split(state["update_key"])

        def epoch_body(carry, epoch_key):
            params, opt_state = carry
            perm = jax.random.permutation(epoch_key, N)[:n_envs_used]

            def gather_envs(idx):
                g = {
                    "obs": trajectory["obs"][:, idx],
                    "action": trajectory["action"][:, idx],
                    "log_prob": trajectory["log_prob"][:, idx],
                    "advantage": advantages[:, idx],
                    "return": returns[:, idx],
                    "done": trajectory["done"][:, idx],
                    "init_carry": init_carry[idx],
                }
                if self.aux_decode_coef > 0.0:
                    g["regime"] = trajectory["regime"][:, idx]
                return g

            shuffled = gather_envs(perm)
            # Custom split because init_carry has a different shape ([N, h])
            # vs trajectory leaves ([T, N, ...]).
            mb_init_carry = shuffled["init_carry"].reshape(
                n_minibatches, mb_size, self.hidden_dim
            )
            mb_traj = jax.tree_util.tree_map(
                lambda x: x.reshape(x.shape[0], n_minibatches, mb_size, *x.shape[2:]),
                {k: v for k, v in shuffled.items() if k != "init_carry"},
            )

            def mb_step(carry, mb_idx):
                params, opt_state = carry
                batch = {k: v[:, mb_idx] for k, v in mb_traj.items()}
                batch["init_carry"] = mb_init_carry[mb_idx]
                (loss, metrics), grads = grad_fn(params, batch)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                metrics["ppo/total_loss"] = loss
                return (params, opt_state), metrics

            (params, opt_state), mb_metrics = jax.lax.scan(
                mb_step, (params, opt_state), jnp.arange(n_minibatches)
            )
            return (params, opt_state), mb_metrics

        epoch_keys = jax.random.split(update_key, self.epochs)
        (params, opt_state), all_metrics = jax.lax.scan(
            epoch_body, (params, opt_state), epoch_keys
        )
        mean_metrics = jax.tree_util.tree_map(lambda x: x.mean(), all_metrics)
        mean_metrics["ppo/advantage_mean"] = advantages.mean()
        mean_metrics["ppo/return_mean"] = returns.mean()
        # Mean L2 norm of the GRU hidden state at end of rollout, averaged
        # over envs. Diagnostic for whether the recurrence is learning a
        # non-trivial representation (vs collapsing to zero).
        mean_metrics["gru/final_carry_norm"] = jnp.linalg.norm(
            final_carry, axis=-1
        ).mean()

        new_state = {
            **state,
            "params": params,
            "opt_state": opt_state,
            "update_key": next_key,
        }
        return new_state, mean_metrics
