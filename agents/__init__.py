"""Agent architectures for meta-RL experiments.

Environment-agnostic model definitions and loss functions.
"""
from .common import MLP, compute_gae, batch_gae
from .ppo import ActorCritic, actor_loss_fn, critic_loss_fn
from .rl2 import GRUActorCritic, rl2_loss_fn
from .rl2_hn import HNActorCritic, apply_generated_policy, rl2_hn_loss_fn, compute_n_policy_params
from .varibad import VariBADActorCritic, elbo_loss_fn, varibad_ppo_loss_fn
