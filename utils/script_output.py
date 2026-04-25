"""Shared-schema script output.

Every script that runs terminates by calling `write_summary(...)` and printing
a single final line. JSON is for Claude Code's decisions; PNG is for humans.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "0.1"

REQUIRED_FIELDS = (
    "script",
    "status",
    "started_at",
    "finished_at",
    "run_mode",
    "config_used",
    "key_stats",
    "outputs_written",
    "error",
    "schema_version",
)

_VALID_STATUSES = ("OK", "FAIL")
_VALID_RUN_MODES = ("super_fast", "fast", "mid", "full", "n/a")


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def validate_summary(summary: dict[str, Any]) -> tuple[bool, str]:
    for field_name in REQUIRED_FIELDS:
        if field_name not in summary:
            return False, f"missing field: {field_name}"
    if summary["status"] not in _VALID_STATUSES:
        return False, f"bad status: {summary['status']}"
    if summary["run_mode"] not in _VALID_RUN_MODES:
        return False, f"bad run_mode: {summary['run_mode']}"
    if not isinstance(summary["outputs_written"], list):
        return False, "outputs_written must be a list"
    if not isinstance(summary["key_stats"], dict):
        return False, "key_stats must be a dict"
    return True, "ok"


def write_summary(
    *,
    script: str,
    status: str,
    run_mode: str,
    started_at: str,
    config_used: dict[str, Any] | None,
    key_stats: dict[str, Any],
    outputs_written: Iterable[str],
    summary_path: str | os.PathLike[str],
    error: str | None = None,
) -> dict[str, Any]:
    """Write the shared-schema summary JSON and return the serialized dict."""
    summary = {
        "script": script,
        "status": status,
        "started_at": started_at,
        "finished_at": iso_now(),
        "run_mode": run_mode,
        "config_used": config_used or {},
        "key_stats": key_stats,
        "outputs_written": list(outputs_written),
        "error": error,
        "schema_version": SCHEMA_VERSION,
    }
    ok, reason = validate_summary(summary)
    if not ok:
        raise ValueError(f"summary does not validate: {reason}")

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def print_ok_line(script: str, *, key_stats: dict[str, Any], output: str) -> None:
    stats_str = " | ".join(f"{k}={v}" for k, v in key_stats.items())
    line = f"[{script}] OK"
    if stats_str:
        line += f" | {stats_str}"
    line += f" | output={output}"
    print(line, flush=True)


def print_fail_line(script: str, *, reason: str) -> None:
    print(f"[{script}] FAIL | reason={reason}", flush=True)


@dataclass
class ScriptRun:
    """Context helper: captures start time, collects outputs, writes summary on exit."""

    script: str
    run_mode: str = "n/a"
    config_used: dict[str, Any] | None = None
    started_at: str = field(default_factory=iso_now)
    outputs: list[str] = field(default_factory=list)

    def add_output(self, path: str | os.PathLike[str]) -> None:
        self.outputs.append(str(path))

    def ok(
        self,
        *,
        key_stats: dict[str, Any],
        summary_path: str | os.PathLike[str],
        print_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        summary = write_summary(
            script=self.script,
            status="OK",
            run_mode=self.run_mode,
            started_at=self.started_at,
            config_used=self.config_used,
            key_stats=key_stats,
            outputs_written=self.outputs + [str(summary_path)],
            summary_path=summary_path,
        )
        print_ok_line(
            self.script,
            key_stats=print_stats if print_stats is not None else key_stats,
            output=str(summary_path),
        )
        return summary

    def fail(
        self,
        *,
        reason: str,
        summary_path: str | os.PathLike[str],
    ) -> dict[str, Any]:
        summary = write_summary(
            script=self.script,
            status="FAIL",
            run_mode=self.run_mode,
            started_at=self.started_at,
            config_used=self.config_used,
            key_stats={},
            outputs_written=self.outputs + [str(summary_path)],
            summary_path=summary_path,
            error=reason,
        )
        print_fail_line(self.script, reason=reason)
        return summary
