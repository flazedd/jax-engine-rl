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
    # When True, the policy and value heads read a stop-gradient copy of the
    # belief (μ, σ), so their gradients do not shape the encoder. With an
    # auxiliary regime-decode loss, the encoder is then shaped only by the
    # VAE + decode objective while a hypernet policy exploits it.
    detach_belief_for_policy: bool = False
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
    # Configurable policy trunk replacing the historical fixed 2-layer MLP at
    # `hidden_dim` width. policy_trunk_layers=2, policy_trunk_hidden=hidden_dim
    # reproduces the previous architecture; per-cell values are set in the
    # locked configs to equalise total parameter counts across the
    # matched-compute factorial.
    policy_trunk_layers: int = 2
    policy_trunk_hidden: int = 0  # 0 = fall back to hidden_dim
    # 3-simplex belief bottleneck: the policy conditions on a learned softmax
    # over regimes instead of the posterior parameters. Off by default.
    belief_simplex_head: bool = False
    n_regimes: int = 3
    # Matched policy inputs. The encoder always consumes the augmented
    # observation [o_t, a_{t-1}, r_{t-1}]; `policy_obs_dim > 0` restricts the
    # policy-side observation to the base observation o_t, as the methodology
    # specifies. Defaults to the legacy behaviour so existing checkpoints load.
    policy_obs_dim: int = 0

    def _policy_obs(self, obs: chex.Array) -> chex.Array:
        if self.policy_obs_dim <= 0:
            return obs
        return obs[..., : self.policy_obs_dim]

    @nn.compact
    def __call__(
        self, obs: chex.Array, mu: chex.Array, log_var: chex.Array
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        if self.detach_belief_for_policy:
            mu = jax.lax.stop_gradient(mu)
            log_var = jax.lax.stop_gradient(log_var)
        sigma = jnp.exp(0.5 * log_var)
        belief = jnp.concatenate([mu, sigma], axis=-1)

        if self.belief_simplex_head:
            simplex_logits = nn.Dense(
                self.n_regimes,
                kernel_init=nn.initializers.orthogonal(0.01),
                name="simplex_head",
            )(belief)
            belief = jax.nn.softmax(simplex_logits)
        else:
            simplex_logits = jnp.zeros((self.n_regimes,), dtype=belief.dtype)

        if self.belief_layernorm and self.integration == "concat" \
                and not self.belief_simplex_head:
            belief_for_concat = nn.LayerNorm()(belief)
        else:
            belief_for_concat = belief
        policy_obs = self._policy_obs(obs)
        x = jnp.concatenate([policy_obs, belief_for_concat], axis=-1)

        trunk_width = self.policy_trunk_hidden or self.hidden_dim
        for _ in range(self.policy_trunk_layers):
            x = nn.Dense(
                trunk_width,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
            )(x)
            x = nn.tanh(x)

        if self.integration == "concat":
            logits = nn.Dense(self.n_actions, kernel_init=nn.initializers.orthogonal(0.01))(x)
        elif self.integration == "hypernet":
            # The belief enters the weight generator directly. A hypernetwork is
            # itself the belief-processing path, so nothing precedes it. The
            # generated target carries the same number of hidden layers as the
            # concat policy, which is what matches the two arms on the
            # observation-to-action map; capacity is equalised by parameter
            # count, not by counting layers on the belief path. The generated
            # weights stay a function of the belief alone, so the observation
            # reaches the action once, through the target.
            hn = Hypernet(
                target_obs_dim=(self.policy_obs_dim or self.obs_size),
                target_hidden=self.hypernet_target_hidden,
                target_output_dim=self.n_actions,
                hypernet_hidden=self.hypernet_hidden,
                init_scale=self.hypernet_init_scale,
                target_hidden_layers=self.policy_trunk_layers,
            )
            flat_weights = hn(belief)
            logits = hn.apply_target(flat_weights, policy_obs)
        else:
            raise ValueError(f"unknown integration: {self.integration!r}")

        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x).squeeze(-1)
        return logits, value, simplex_logits


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
    # Policy objective: "ppo" (clipped surrogate, default) or "a2c" (vanilla
    # advantage actor-critic, no importance ratio and no clipping). Used to
    # test whether the integration finding is specific to PPO; A2C should be
    # run with epochs=1 to stay on-policy.
    policy_objective: str = "ppo"
    reward_decoder: str = "bernoulli"  # {"bernoulli", "gaussian"}
    # Integration mechanism: "concat" = MLP([obs,μ,σ])→logits;
    # "hypernet" = belief=[μ,σ] generates target-MLP weights, target maps obs→logits.
    integration: str = "concat"
    hypernet_target_hidden: int = 16
    hypernet_hidden: int = 64
    # Architectural-care knobs (defaults preserve previous behaviour).
    belief_layernorm: bool = False
    hypernet_init_scale: float = 0.01
    # Matched-compute knobs: policy trunk between [obs, belief] and the
    # action / value heads. Defaults reproduce the historical 2-layer MLP
    # at `hidden_dim` width.
    policy_trunk_layers: int = 2
    policy_trunk_hidden: int = 0  # 0 = fall back to hidden_dim

    # Single-stage "best of both" knob: an auxiliary regime-decode loss on the
    # latent mean μ, added to the loss with this coefficient (0 = off, no aux
    # params, behaviour identical to before). Pressures the encoder to keep the
    # regime linearly decodable from μ while a hypernet policy uses the belief.
    aux_decode_coef: float = 0.0
    n_regimes: int = 3
    # Detach the belief (μ, σ) into the policy/value heads (single-stage
    # decoupling); only meaningful together with aux_decode_coef > 0.
    detach_belief_for_policy: bool = False
    # 3-simplex belief bottleneck (see VariBADPolicy). When on, the policy
    # conditions on a learned softmax over regimes; with aux_decode_coef > 0 that
    # softmax is supervised toward the true regime. Off by default.
    belief_simplex_head: bool = False
    # Matched policy inputs; see VariBADPolicy for the semantics.
    policy_obs_dim: int = 0

    # ---- Reconstruction target ------------------------------------------
    # `recon_window = 0` reproduces the original single-latent objective: one
    # sample from the final-step posterior q(m | tau_{:T}) reconstructs every
    # reward in the trajectory. That assumes one task per episode.
    #
    # `recon_window = K > 0` is the filtering form used when the task changes
    # within the episode. For every step t a sample is drawn from q(m | tau_{:t})
    # and asked to predict only the next K rewards, r_t .. r_{t+K-1}. The
    # regime is roughly constant over a window much shorter than its dwell
    # time, so the target is consistent with the latent that has to explain it,
    # and every posterior is scored rather than only the last one.
    recon_window: int = 0
    # Width of the observation slice the decoder reads. 0 = the whole augmented
    # tuple. Set to the base observation width so the decoder does not receive
    # the previous reward: rewards are correlated through a persistent regime,
    # so a decoder given r_{t-1} can predict r_t without consulting the latent,
    # which weakens the very gradient the reconstruction exists to provide.
    decoder_obs_dim: int = 0

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

    def _decoder_obs(self, obs: chex.Array) -> chex.Array:
        """Slice the decoder's view of the observation. See `decoder_obs_dim`."""
        if self.decoder_obs_dim <= 0:
            return obs
        return obs[..., : self.decoder_obs_dim]

    def _policy(self) -> VariBADPolicy:
        return VariBADPolicy(
            policy_obs_dim=self.policy_obs_dim,
            n_actions=self.n_actions,
            obs_size=self.obs_size,
            latent_dim=self.latent_dim,
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
        dummy_obs_base = jnp.zeros(
            (self.decoder_obs_dim or self.obs_size,), dtype=jnp.float32
        )
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
        if self.aux_decode_coef > 0.0 and not self.belief_simplex_head:
            # Linear regime decoder over the latent mean μ; trained jointly as a
            # sibling collection so the sub-modules (which read their own keys)
            # are untouched while the optimizer still updates it.
            k_aux = jax.random.fold_in(key, 99)
            params["aux"] = {
                "W": nn.initializers.orthogonal(1.0)(
                    k_aux, (self.latent_dim, self.n_regimes)),
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
        new_carry, mu, log_var = self._encoder().apply(
            state["params"]["encoder"], carry, obs
        )
        logits, value, simplex_logits = self._policy().apply(
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
        if self.belief_simplex_head:
            extras["belief_simplex"] = jax.nn.softmax(simplex_logits)
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
            _, v, _ = policy.apply(params["policy"], obs_i, mu_i, lv_i)
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
            logits, values, simplex_logits = jax.vmap(jax.vmap(policy_apply))(
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
            if self.policy_objective == "a2c":
                # Vanilla advantage actor-critic: plain policy-gradient, no
                # importance ratio and no clipping (run with epochs=1).
                policy_loss = -(lp_new * adv).mean()
            else:
                unclipped = ratio * adv
                clipped = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -jnp.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * jnp.mean((values - batch["return"]) ** 2)
            probs = jnp.exp(log_probs_all)
            entropy = -jnp.sum(probs * log_probs_all, axis=-1).mean()
            ppo_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            # 4. VAE loss. See `recon_window` for the two forms.
            decoder = self._decoder()
            decode_apply = lambda mm, oo, aa: decoder.apply(
                params["decoder"], mm, oo, aa
            )
            dec_obs = self._decoder_obs(batch["obs"])   # [T, n_envs, dec_dim]
            sigmas = jnp.exp(0.5 * log_vars)

            if self.recon_window <= 0:
                # Original: one sample from q(m | tau_{:T}) explains everything.
                eps = jax.random.normal(sample_key, mus[-1].shape)
                m = mus[-1] + sigmas[-1] * eps          # [n_envs, latent_dim]
                m_bcast = jnp.broadcast_to(m[None, :, :], (T,) + m.shape)
                r_pred = jax.vmap(jax.vmap(decode_apply))(
                    m_bcast, dec_obs, batch["action"]
                )                                       # [T, n_envs]
                r_target = batch["reward"]
                weight = jnp.ones_like(r_target)
            else:
                # Filtering form: q(m | tau_{:t}) predicts the next K rewards.
                K = int(self.recon_window)
                eps = jax.random.normal(sample_key, mus.shape)
                m_t = mus + sigmas * eps                # [T, n_envs, latent_dim]

                # Target index t+k, clamped so the gather stays in range; the
                # mask zeroes the steps that ran past the end of the rollout,
                # keeping shapes static for jit.
                t_idx = jnp.arange(T)[:, None]          # [T, 1]
                k_idx = jnp.arange(K)[None, :]          # [1, K]
                tau = jnp.minimum(t_idx + k_idx, T - 1)  # [T, K]
                weight = (t_idx + k_idx < T).astype(jnp.float32)[..., None]

                m_rep = jnp.broadcast_to(
                    m_t[:, None, :, :], (T, K) + m_t.shape[1:]
                )                                       # [T, K, n_envs, latent]
                obs_win = dec_obs[tau]                  # [T, K, n_envs, dec_dim]
                act_win = batch["action"][tau]          # [T, K, n_envs]
                r_target = batch["reward"][tau]         # [T, K, n_envs]
                weight = jnp.broadcast_to(weight, r_target.shape)

                r_pred = jax.vmap(jax.vmap(jax.vmap(decode_apply)))(
                    m_rep, obs_win, act_win
                )                                       # [T, K, n_envs]

            denom = jnp.maximum(weight.sum(), 1.0)
            if self.reward_decoder == "bernoulli":
                per_step = optax.sigmoid_binary_cross_entropy(r_pred, r_target)
                recon_loss = (per_step * weight).sum() / denom
            elif self.reward_decoder == "gaussian":
                per_step = 0.5 * (r_pred - r_target) ** 2
                recon_loss = (per_step * weight).sum() / denom
            else:
                raise ValueError(
                    f"unknown reward_decoder: {self.reward_decoder!r}"
                )

            # Per-coordinate KL(q(m | τ_{:t}) || N(0, I)), averaged over
            # (time, environment, latent coordinate). The coefficient on the
            # full multivariate KL is kl_coef / latent_dim.
            kl = -0.5 * jnp.mean(
                1.0 + log_vars - mus ** 2 - jnp.exp(log_vars)
            )

            vae_loss = recon_loss + self.kl_coef * kl

            total_loss = ppo_loss + self.vae_coef * vae_loss

            # Single-stage auxiliary regime-decode on the differentiable latent
            # mean μ: gradient flows into the encoder, pressuring it to keep the
            # regime decodable while the policy uses the (detached) belief.
            aux_loss = jnp.array(0.0)
            aux_acc = jnp.array(0.0)
            if self.aux_decode_coef > 0.0:
                # With a simplex head the regime cross-entropy supervises the
                # softmax the policy conditions on; otherwise it trains a
                # separate linear decoder over the latent mean.
                if self.belief_simplex_head:
                    aux_logits = simplex_logits
                else:
                    aux_logits = (
                        jnp.einsum("tnl,lr->tnr", mus, params["aux"]["W"])
                        + params["aux"]["b"]
                    )
                aux_loss = optax.softmax_cross_entropy_with_integer_labels(
                    aux_logits, batch["regime"]
                ).mean()
                aux_acc = (aux_logits.argmax(-1) == batch["regime"]).mean()
                total_loss = total_loss + self.aux_decode_coef * aux_loss

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
                "aux/decode_loss": aux_loss,
                "aux/decode_acc": aux_acc,
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
                g = {
                    "obs": trajectory["obs"][:, idx],
                    "action": trajectory["action"][:, idx],
                    "reward": trajectory["reward"][:, idx],
                    "log_prob": trajectory["log_prob"][:, idx],
                    "advantage": advantages[:, idx],
                    "return": returns[:, idx],
                    "done": trajectory["done"][:, idx],
                    "init_carry": init_carry[idx],
                }
                if self.aux_decode_coef > 0.0:
                    g["regime"] = trajectory["regime"][:, idx]
                return g

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
