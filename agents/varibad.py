"""VariBAD (Zintgraf et al., 2020) — variational Bayes-adaptive deep RL.

Architecture:
- **Encoder** (GRU): consumes the augmented observation
  `[obs, prev_action, prev_reward, prev_done]` and emits
  `(μ_t, log σ²_t)` at every step — the running variational posterior over
  the latent task `m`.
- **Policy** (MLP): conditions on `(obs_t, μ_t, σ_t)` to produce
  `(logits, value)`. Non-recurrent — the encoder carries all history.
- **Reward decoder** (MLP): given a sampled `m` and `(obs_τ, action_τ)`,
  predicts a scalar that the VAE reconstruction loss compares against
  `r_τ`. Two heads are supported, controlled by `reward_decoder`:
    - `"bernoulli"` (default): output is a logit; loss is sigmoid BCE
      against `r_τ ∈ {0, 1}`. Use for bandit / gridworld / regime_bandit.
    - `"gaussian"`: output is the predicted mean μ; loss is `0.5·(μ − r)²`
      (fixed-σ Gaussian NLL up to a constant). Use for continuous-reward
      envs (MM).

Loss:
- **PPO** on the policy (using online posterior).
- **VAE / ELBO**: at trajectory end, sample `m ~ q(m | τ_{:T})`. Decode all
  rewards `r_τ` for τ ∈ [0, T) from `(m, obs_τ, action_τ)` and compute the
  recon loss for the configured head. KL-regularize the running posterior
  `q(m | τ_{:t})` toward `N(0, I)`, averaged over t.

Encoder hidden state resets on episode boundaries (each episode = one task).
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


class VariBADEncoder(nn.Module):
    hidden_dim: int = 64
    latent_dim: int = 8

    @nn.compact
    def __call__(
        self, carry: chex.Array, obs_aug: chex.Array
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
        )(obs_aug)
        x = nn.tanh(x)
        new_carry, _ = nn.GRUCell(features=self.hidden_dim)(carry, x)
        head = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))
        )(new_carry)
        head = nn.tanh(head)
        mu = nn.Dense(
            self.latent_dim, kernel_init=nn.initializers.orthogonal(0.01)
        )(head)
        log_var = nn.Dense(
            self.latent_dim, kernel_init=nn.initializers.orthogonal(0.01)
        )(head)
        # Clamp log_var for numerical stability.
        log_var = jnp.clip(log_var, -10.0, 2.0)
        return new_carry, mu, log_var


class VariBADRewardDecoder(nn.Module):
    hidden_dim: int = 64
    n_actions: int = 2

    @nn.compact
    def __call__(self, m: chex.Array, obs: chex.Array, action: chex.Array) -> chex.Array:
        a_oh = jax.nn.one_hot(action.astype(jnp.int32), self.n_actions, dtype=jnp.float32)
        x = jnp.concatenate([m, obs, a_oh], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.tanh(x)
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(0.01))(x).squeeze(-1)


class VariBADPolicy(nn.Module):
    """VariBAD policy + value head.

    Two integration paths, selected by `integration`:
    - "concat" (default): policy is MLP on [obs, μ, σ].
    - "hypernet": belief = [μ, σ] generates the weights of a small target
      MLP via Hypernet; target operates on raw obs to produce logits.
      Value head stays MLP on [obs, μ, σ] regardless.
    """

    n_actions: int
    obs_size: int
    latent_dim: int
    hidden_dim: int = 64
    integration: str = "concat"
    hypernet_target_hidden: int = 16
    hypernet_hidden: int = 64
    # Concat-side architectural-care knob: LayerNorm the [μ, σ] belief vector
    # before concatenating with obs. No-op for the hypernet path. Default
    # False preserves prior behaviour.
    belief_layernorm: bool = False
    # Hypernet-side architectural-care knob: init scale on the hypernet output
    # layer. 0.0 = Beck et al. zero-init. Default 0.01 preserves prior code.
    hypernet_init_scale: float = 0.01

    @nn.compact
    def __call__(
        self, obs: chex.Array, mu: chex.Array, log_var: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        sigma = jnp.exp(0.5 * log_var)
        belief = jnp.concatenate([mu, sigma], axis=-1)
        if self.belief_layernorm and self.integration == "concat":
            belief_for_concat = nn.LayerNorm()(belief)
        else:
            belief_for_concat = belief
        x = jnp.concatenate([obs, belief_for_concat], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.tanh(x)

        if self.integration == "concat":
            logits = nn.Dense(self.n_actions, kernel_init=nn.initializers.orthogonal(0.01))(x)
        elif self.integration == "hypernet":
            hn = Hypernet(
                target_obs_dim=self.obs_size,
                target_hidden=self.hypernet_target_hidden,
                target_output_dim=self.n_actions,
                hypernet_hidden=self.hypernet_hidden,
                init_scale=self.hypernet_init_scale,
            )
            flat_weights = hn(belief)
            logits = hn.apply_target(flat_weights, obs)
        else:
            raise ValueError(f"unknown integration: {self.integration!r}")

        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x).squeeze(-1)
        return logits, value


@dataclass(frozen=True)
class VariBADAgent:
    obs_size: int           # augmented obs size (from RL2ObsEnv: base + n_actions + 2)
    n_actions: int
    hidden_dim: int = 64
    latent_dim: int = 8
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    vae_coef: float = 1.0       # weight on full VAE loss (recon + KL)
    kl_coef: float = 0.1        # weight on KL term inside the VAE loss
    gamma: float = 0.99
    lam: float = 0.95
    epochs: int = 4
    minibatch_envs: int = 32
    reward_decoder: str = "bernoulli"  # {"bernoulli", "gaussian"}
    # Integration mechanism: "concat" = MLP([obs,μ,σ])→logits;
    # "hypernet" = belief=[μ,σ] generates target-MLP weights, target maps obs→logits.
    integration: str = "concat"
    hypernet_target_hidden: int = 16
    hypernet_hidden: int = 64
    # Architectural-care knobs (defaults preserve previous behaviour).
    belief_layernorm: bool = False
    hypernet_init_scale: float = 0.01

    requires_regime_label: bool = False
    requires_analytical_posterior: bool = False
    is_recurrent: bool = True
    produces_belief_for_eval: bool = False
    # Trajectory key carrying this method's belief vector. Used by the
    # posterior-quality probe (evaluation/posterior_probe.py) to identify
    # the agent's internal belief representation.
    belief_key: str = "mu"

    # ---- factories ------------------------------------------------------

    def _encoder(self) -> VariBADEncoder:
        return VariBADEncoder(hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)

    def _decoder(self) -> VariBADRewardDecoder:
        return VariBADRewardDecoder(hidden_dim=self.hidden_dim, n_actions=self.n_actions)

    def _policy(self) -> VariBADPolicy:
        return VariBADPolicy(
            n_actions=self.n_actions,
            obs_size=self.obs_size,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            integration=self.integration,
            hypernet_target_hidden=self.hypernet_target_hidden,
            hypernet_hidden=self.hypernet_hidden,
            belief_layernorm=self.belief_layernorm,
            hypernet_init_scale=self.hypernet_init_scale,
        )

    def _optimizer(self) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.learning_rate),
        )

    # ---- Agent API ------------------------------------------------------

    def init(self, key: chex.PRNGKey) -> chex.ArrayTree:
        k_e, k_d, k_p = jax.random.split(key, 3)
        dummy_carry = jnp.zeros((self.hidden_dim,), dtype=jnp.float32)
        dummy_obs_aug = jnp.zeros((self.obs_size,), dtype=jnp.float32)
        dummy_obs_base = jnp.zeros((self.obs_size,), dtype=jnp.float32)
        dummy_m = jnp.zeros((self.latent_dim,), dtype=jnp.float32)
        dummy_action = jnp.asarray(0, dtype=jnp.int32)
        dummy_log_var = jnp.zeros((self.latent_dim,), dtype=jnp.float32)

        encoder_params = self._encoder().init(k_e, dummy_carry, dummy_obs_aug)
        decoder_params = self._decoder().init(k_d, dummy_m, dummy_obs_base, dummy_action)
        policy_params = self._policy().init(
            k_p, dummy_obs_aug, dummy_m, dummy_log_var
        )

        params = {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "policy": policy_params,
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
        new_carry, mu, log_var = self._encoder().apply(
            state["params"]["encoder"], carry, obs
        )
        logits, value = self._policy().apply(
            state["params"]["policy"], obs, mu, log_var
        )
        action = jax.random.categorical(key, logits)
        log_prob = jax.nn.log_softmax(logits)[action]
        extras = {
            "log_prob": log_prob,
            "value": value,
            "mu": mu,
            "log_var": log_var,
        }
        return action, extras, new_carry

    # ---- update helpers -------------------------------------------------

    def _replay_encoder(self, encoder_params, init_carry, obs_seq, done_seq):
        """Per-env replay of the encoder forward pass. Returns mus, log_vars
        of shape [T, latent_dim] each, plus the final carry."""
        encoder = self._encoder()

        def step(carry, inputs):
            obs_t, done_t = inputs
            new_carry, mu_t, log_var_t = encoder.apply(encoder_params, carry, obs_t)
            zeros = jnp.zeros_like(new_carry)
            mask = done_t.astype(new_carry.dtype)
            next_carry = mask * zeros + (1.0 - mask) * new_carry
            return next_carry, (mu_t, log_var_t)

        final_carry, (mus, log_vars) = jax.lax.scan(
            step, init_carry, (obs_seq, done_seq)
        )
        return mus, log_vars, final_carry

    def _bootstrap_value(self, params, final_obs, final_carry):
        """V(final_obs | running posterior) per env for GAE bootstrap."""
        encoder, policy = self._encoder(), self._policy()
        def f(carry_i, obs_i):
            new_c, mu_i, lv_i = encoder.apply(params["encoder"], carry_i, obs_i)
            _, v = policy.apply(params["policy"], obs_i, mu_i, lv_i)
            return v
        return jax.vmap(f)(final_carry, final_obs)

    # ---- update ---------------------------------------------------------

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

        T, N = trajectory["reward"].shape
        mb_size = min(self.minibatch_envs, N)
        n_minibatches = max(1, N // mb_size)
        n_envs_used = n_minibatches * mb_size

        update_key, next_key = jax.random.split(state["update_key"])

        def loss_fn(params, batch, sample_key):
            # 1. Replay encoder per-env over T steps.
            replay = jax.vmap(
                lambda c, o, d: self._replay_encoder(params["encoder"], c, o, d),
                in_axes=(0, 1, 1), out_axes=(1, 1, 0),
            )
            mus, log_vars, _ = replay(
                batch["init_carry"], batch["obs"], batch["done"]
            )
            # mus: [T, n_envs, latent_dim], log_vars: [T, n_envs, latent_dim]

            # 2. Policy logits/values from (obs_t, mu_t, log_var_t).
            policy = self._policy()
            policy_apply = lambda o, m, lv: policy.apply(
                params["policy"], o, m, lv
            )
            logits, values = jax.vmap(jax.vmap(policy_apply))(
                batch["obs"], mus, log_vars
            )
            # logits [T, n_envs, n_actions], values [T, n_envs]

            # 3. PPO loss.
            log_probs_all = jax.nn.log_softmax(logits)
            lp_new = jnp.take_along_axis(
                log_probs_all, batch["action"][..., None], axis=-1
            ).squeeze(-1)
            ratio = jnp.exp(lp_new - batch["log_prob"])
            adv = batch["advantage"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            unclipped = ratio * adv
            clipped = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            policy_loss = -jnp.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * jnp.mean((values - batch["return"]) ** 2)
            probs = jnp.exp(log_probs_all)
            entropy = -jnp.sum(probs * log_probs_all, axis=-1).mean()
            ppo_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            # 4. VAE loss: sample m from final posterior q(m | τ_{:T}) and
            #    reconstruct ALL rewards r_τ for τ ∈ [0, T).
            mu_T = mus[-1]              # [n_envs, latent_dim]
            log_var_T = log_vars[-1]
            sigma_T = jnp.exp(0.5 * log_var_T)
            eps = jax.random.normal(sample_key, mu_T.shape)
            m = mu_T + sigma_T * eps    # [n_envs, latent_dim]

            decoder = self._decoder()
            # Decode r_τ for every τ; broadcast m over T.
            m_bcast = jnp.broadcast_to(m[None, :, :], (T,) + m.shape)
            decode_apply = lambda mm, oo, aa: decoder.apply(
                params["decoder"], mm, oo, aa
            )
            r_logits = jax.vmap(jax.vmap(decode_apply))(
                m_bcast, batch["obs"], batch["action"]
            )  # [T, n_envs]

            r_target = batch["reward"]
            if self.reward_decoder == "bernoulli":
                recon_loss = optax.sigmoid_binary_cross_entropy(
                    r_logits, r_target
                ).mean()
            elif self.reward_decoder == "gaussian":
                recon_loss = 0.5 * jnp.mean((r_logits - r_target) ** 2)
            else:
                raise ValueError(
                    f"unknown reward_decoder: {self.reward_decoder!r}"
                )

            # KL(q(m | τ_{:t}) || N(0, I)) averaged over (t, env). Standard
            # closed-form for diagonal Gaussian vs unit Gaussian.
            kl = -0.5 * jnp.mean(
                1.0 + log_vars - mus ** 2 - jnp.exp(log_vars)
            )

            vae_loss = recon_loss + self.kl_coef * kl

            total_loss = ppo_loss + self.vae_coef * vae_loss

            approx_kl = (batch["log_prob"] - lp_new).mean()
            clipped_frac = (jnp.abs(ratio - 1.0) > self.clip_eps).astype(
                jnp.float32
            ).mean()
            metrics = {
                "ppo/policy_loss": policy_loss,
                "ppo/value_loss": value_loss,
                "ppo/entropy": entropy,
                "ppo/approx_kl": approx_kl,
                "ppo/clipped_frac": clipped_frac,
                "ppo/total_loss": ppo_loss,
                "vae/recon_loss": recon_loss,
                "vae/kl": kl,
                "vae/total": vae_loss,
                "total_loss": total_loss,
                "posterior/mu_norm": jnp.linalg.norm(mus, axis=-1).mean(),
                "posterior/sigma_mean": jnp.exp(0.5 * log_vars).mean(),
            }
            return total_loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        optimizer = self._optimizer()

        def epoch_body(carry, epoch_key):
            params, opt_state = carry
            shuf_key, sample_key = jax.random.split(epoch_key)
            perm = jax.random.permutation(shuf_key, N)[:n_envs_used]

            def gather(idx):
                return {
                    "obs": trajectory["obs"][:, idx],
                    "action": trajectory["action"][:, idx],
                    "reward": trajectory["reward"][:, idx],
                    "log_prob": trajectory["log_prob"][:, idx],
                    "advantage": advantages[:, idx],
                    "return": returns[:, idx],
                    "done": trajectory["done"][:, idx],
                    "init_carry": init_carry[idx],
                }

            shuffled = gather(perm)
            mb_init_carry = shuffled["init_carry"].reshape(
                n_minibatches, mb_size, self.hidden_dim
            )
            mb_traj = jax.tree_util.tree_map(
                lambda x: x.reshape(x.shape[0], n_minibatches, mb_size, *x.shape[2:]),
                {k: v for k, v in shuffled.items() if k != "init_carry"},
            )
            mb_sample_keys = jax.random.split(sample_key, n_minibatches)

            def mb_step(carry, mb_inp):
                params, opt_state = carry
                mb_idx, sk = mb_inp
                batch = {k: v[:, mb_idx] for k, v in mb_traj.items()}
                batch["init_carry"] = mb_init_carry[mb_idx]
                (loss, metrics), grads = grad_fn(params, batch, sk)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                metrics["total_loss"] = loss
                return (params, opt_state), metrics

            (params, opt_state), mb_metrics = jax.lax.scan(
                mb_step, (params, opt_state), (jnp.arange(n_minibatches), mb_sample_keys)
            )
            return (params, opt_state), mb_metrics

        epoch_keys = jax.random.split(update_key, self.epochs)
        (params, opt_state), all_metrics = jax.lax.scan(
            epoch_body, (params, opt_state), epoch_keys
        )
        mean_metrics = jax.tree_util.tree_map(lambda x: x.mean(), all_metrics)
        mean_metrics["ppo/advantage_mean"] = advantages.mean()
        mean_metrics["ppo/return_mean"] = returns.mean()

        new_state = {
            **state,
            "params": params,
            "opt_state": opt_state,
            "update_key": next_key,
        }
        return new_state, mean_metrics
