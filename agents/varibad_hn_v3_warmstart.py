"""V3: Warm-Start — ELBO pre-trains the posterior before PPO begins.

The posterior starts at N(0,1) due to zero-init, so the HyperNet initially
sees uninformative μ/σ and learns to ignore them. By the time the posterior
becomes meaningful, the HN has found a solution that doesn't use it.

Fix: run WARMSTART_ITERS of ELBO-only updates so the posterior already
carries task/regime information when PPO starts. The training loop should
skip PPO updates for the first WARMSTART_ITERS iterations.

Uses the same VariBADHNActorCritic model and loss functions.
"""
from .varibad_hn import (VariBADHNActorCritic, varibad_hn_elbo_loss_fn,
                          varibad_hn_ppo_loss_fn)

# Re-export model and losses
V3WarmStartActorCritic = VariBADHNActorCritic
v3_warmstart_ppo_loss_fn = varibad_hn_ppo_loss_fn
v3_warmstart_elbo_loss_fn = varibad_hn_elbo_loss_fn

# Training scripts should skip PPO for the first WARMSTART_ITERS iterations
WARMSTART_ITERS = 50
