from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ENV_PREFIXES = ("CUDA_", "TORCH_", "HF_", "TRANSFORMERS_", "WANDB_", "MODAL_", "PYTHON")


@dataclass(frozen=True)
class RunDir:
    path: Path
    run_id: str

    @property
    def config(self) -> Path:
        return self.path / "config.json"

    @property
    def metrics(self) -> Path:
        return self.path / "metrics.jsonl"

    @property
    def env(self) -> Path:
        return self.path / "env.json"

    @property
    def pip_freeze(self) -> Path:
        return self.path / "pip_freeze.txt"

    @property
    def checkpoints(self) -> Path:
        return self.path / "checkpoints"


def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.rstrip("\n") or None


def _pip_freeze() -> str:
    out = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout


def _env_snapshot() -> dict[str, str]:
    return {k: os.environ[k] for k in sorted(os.environ) if k.startswith(_ENV_PREFIXES)}


def create_run_dir(
    base: Path | str,
    *,
    run_id: str | None = None,
    suffix: str | None = None,
    config: dict[str, Any] | None = None,
) -> RunDir:
    """Create ``<base>/<run_id>/`` and snapshot environment, git, and config.

    The run dir contains:
      - ``env.json``       — interpreter, platform, git SHA/dirty, argv, env-var subset.
      - ``pip_freeze.txt`` — output of ``pip freeze`` at run start.
      - ``config.json``    — caller-supplied config (only if ``config`` is not None).
      - ``metrics.jsonl``  — empty file; appenders write one JSON object per line.
      - ``checkpoints/``   — empty dir for adapter/optimizer state.

    Raises ``FileExistsError`` if the run dir already exists. Pick a fresh ``run_id``
    on retry rather than overwriting silently — old runs are cheap to keep.
    """
    base = Path(base)
    if run_id is None:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        if suffix:
            # Disambiguate runs created in the same second (e.g. matrix fan-out)
            # and make the dir self-describing: <timestamp>_<cell-label>.
            run_id = f"{run_id}_{suffix}"
    path = base / run_id
    if path.exists():
        raise FileExistsError(f"run dir already exists: {path}")
    path.mkdir(parents=True)
    (path / "checkpoints").mkdir()

    snapshot = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "executable": sys.executable,
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "argv": sys.argv,
        "env": _env_snapshot(),
    }
    (path / "env.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    (path / "pip_freeze.txt").write_text(_pip_freeze())
    (path / "metrics.jsonl").touch()
    if config is not None:
        (path / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    return RunDir(path=path, run_id=run_id)
