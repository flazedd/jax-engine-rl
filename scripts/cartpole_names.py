"""Single source of the cartpole experiment-directory naming.

The sweep orchestrator writes these directories and the probe and hypothesis
tests read them, so the scheme lives in one module: three copies of the rule
drifted apart once already, which silently pointed the analyses at runs the
sweep had not produced.

Names are prefixed `m_cartpole_matched_`. The pre-matched runs, where the
references were memoryless on o_t and ran different optimiser settings, keep
the unprefixed `m_cartpole_<method>` paths and are retained as history only.
"""
from __future__ import annotations

PREFIX = "m_cartpole_matched"


def cartpole_experiment_name(method: str, axis: str, level: str) -> str:
    """Directory under `results/` for one (method, axis, level) cell.

    Medium is shared across axes: the environment is identical there, so the
    cell is trained once and the axis is omitted from the name.
    """
    if level == "medium":
        return f"{PREFIX}_{method}"
    return f"{PREFIX}_{method}_{axis}_{level}"
