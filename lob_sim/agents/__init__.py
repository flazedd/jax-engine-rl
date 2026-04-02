"""Agent registry.

All agents follow the Agent protocol from base.py.  Use make_agent() to
instantiate by name — this is the single entry point used by training scripts.
"""
from lob_sim.agents.base import Agent, AgentState, RolloutBatch

# Registry: name → constructor(key, **config_overrides) → (agent, config)
_REGISTRY: dict[str, callable] = {}


def _build_ppo(key, **overrides):
    from lob_sim.agents.ppo import PPOAgent, PPOConfig
    cfg = PPOConfig(**overrides) if overrides else PPOConfig()
    return PPOAgent(ppo_config=cfg, key=key), cfg


# Phase 7+: add builders as agents are implemented
# def _build_rl2(key, **overrides):
#     from lob_sim.agents.rl2 import RL2Agent, RL2Config
#     cfg = RL2Config(**overrides) if overrides else RL2Config()
#     return RL2Agent(rl2_config=cfg, key=key), cfg


_REGISTRY["ppo"] = _build_ppo
# _REGISTRY["rl2"] = _build_rl2
# _REGISTRY["varibad"] = _build_varibad
# _REGISTRY["varibad_hyper"] = _build_varibad_hyper


def available_agents() -> list[str]:
    """Return names of all registered agents."""
    return list(_REGISTRY.keys())


def make_agent(name: str, *, key, **config_overrides):
    """Instantiate an agent by name.

    Returns (agent, config) where config is the agent's hyperparameter
    NamedTuple.  Keyword arguments override default config fields.

    Usage:
        agent, cfg = make_agent("ppo", key=key, lr=1e-3, n_envs=32)
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown agent {name!r}. "
                         f"Available: {available_agents()}")
    return _REGISTRY[name](key, **config_overrides)
