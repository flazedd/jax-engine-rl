"""V30: Decision-Theoretic Abstain Gate — V2 + explicit abstain head.

V2 emits a flat categorical over K+1 actions. V30 factors the decision:
  arm-selector head (HN-generated MLP): obs -> K arm logits
  abstain-gate head (small MLP):        (h, μ) -> p_abstain (sigmoid)

Combined categorical: P(a_k) = arm_probs[k] · (1 - p_abstain)  for k < K
                     P(abstain)   = p_abstain

This architectural decomposition matches the Bayes-optimal decision rule:
pick MAP arm unless expected reward below abstain_payoff. The gate head
only has to learn calibrated "should I commit" probabilities; the arm head
only has to learn "which arm". Cleaner loss landscape than a flat K+1 softmax.

ELBO is identical to V2:MuOnly.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from .common import MLP, ppo_loss_from_logits
from .rl2_hn import apply_generated_policy, compute_n_policy_params
from .varibad_hn import varibad_hn_elbo_loss_fn


v30_elbo_loss_fn = varibad_hn_elbo_loss_fn


class V30AbstainGateActorCritic(eqx.Module):
    gru_cell: eqx.nn.GRUCell
    posterior_net: MLP
    decoder_net: MLP
    hyper_net: MLP              # generates K-arm policy (no abstain)
    abstain_gate: MLP           # (h, μ) -> scalar logit
    critic_head: MLP
    hidden_size: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    policy_hidden: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)
    n_arms: int = eqx.field(static=True)

    def __init__(self, input_size, obs_size, n_actions,
                 hidden_size=64, latent_dim=4, policy_hidden=16, *, key):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.obs_size = obs_size
        self.policy_hidden = policy_hidden
        self.n_actions = n_actions
        self.n_arms = n_actions - 1  # last action index is "abstain"

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, key=k1)

        post = MLP([hidden_size, 64, 2 * latent_dim], key=k2)
        self.posterior_net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), post,
            (jnp.zeros_like(post.layers[-1].weight),
             jnp.zeros_like(post.layers[-1].bias)))

        decoder_input = latent_dim + obs_size + n_actions
        self.decoder_net = MLP([decoder_input, 64, 1 + obs_size], key=k3)

        hn_input = hidden_size + latent_dim
        # HN generates an n_arms-way policy (no abstain action)
        n_params = compute_n_policy_params(obs_size, policy_hidden, self.n_arms)
        hn = MLP([hn_input, 256, n_params], key=k4)
        self.hyper_net = eqx.tree_at(
            lambda m: m.layers[-1].weight, hn,
            jnp.zeros_like(hn.layers[-1].weight))

        # Abstain gate on (h, μ) — bias init to abstain often at start
        gate = MLP([hidden_size + latent_dim, 32, 1], key=k5)
        self.abstain_gate = eqx.tree_at(
            lambda m: m.layers[-1].weight, gate,
            jnp.zeros_like(gate.layers[-1].weight))

        critic_input = hidden_size + 2 * latent_dim + obs_size
        self.critic_head = MLP([critic_input, 64, 1], key=k6)

    def _logits_and_value(self, aug_input, obs, hidden):
        new_h = self.gru_cell(aug_input, hidden)
        post = self.posterior_net(new_h)
        mu = post[:self.latent_dim]
        log_sig = jnp.clip(post[self.latent_dim:], -5.0, 2.0)
        sigma = jnp.exp(log_sig)

        hn_in = jnp.concatenate([new_h, mu])
        policy_params = self.hyper_net(hn_in)
        arm_logits = apply_generated_policy(
            obs, policy_params, self.obs_size, self.policy_hidden, self.n_arms)
        gate_logit = self.abstain_gate(jnp.concatenate([new_h, mu])).squeeze(-1)

        # Combine into K+1 categorical in log-space for numerical stability
        # P(arm_k) = softmax(arm_logits)_k · (1 - p_abstain)
        # P(abstain) = p_abstain
        log_arm = jax.nn.log_softmax(arm_logits)                  # (K,)
        log_p_abst = jax.nn.log_sigmoid(gate_logit)               # log p
        log_1m_p_abst = jax.nn.log_sigmoid(-gate_logit)           # log (1-p)
        combined_log = jnp.concatenate(
            [log_arm + log_1m_p_abst, log_p_abst[None]])          # (K+1,)

        critic_in = jnp.concatenate([new_h, mu, sigma, obs])
        value = self.critic_head(critic_in).squeeze(-1)
        return combined_log, value, new_h

    def forward_step(self, aug_input, obs, hidden):
        return self._logits_and_value(aug_input, obs, hidden)


def v30_ppo_loss_fn(model, batch, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
    obs, prev_act, prev_rew, gru_h, actions, old_lp, advantages, returns = batch
    aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)

    def forward_one(x, o, h):
        log_probs, value, _ = model._logits_and_value(x, o, h)
        return log_probs, value

    log_probs_all, values = jax.vmap(forward_one)(aug, obs, gru_h)
    # ppo_loss_from_logits applies log_softmax internally, which is idempotent
    # on already-normalized log-probs. Use the log-probs directly as "logits".
    return ppo_loss_from_logits(
        log_probs_all, values, actions, old_lp, advantages, returns,
        clip_eps, ent_coef, vf_coef)
