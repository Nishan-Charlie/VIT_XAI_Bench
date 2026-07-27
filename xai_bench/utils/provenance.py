"""Environment and code provenance captured with every benchmark run.

The goal is that any number in the benchmark can be traced back to the exact
code, environment and configuration that produced it. Values that cannot be
determined are recorded as ``None`` — never guessed.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    """Run a git command in the repo, returning stripped stdout or None."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_commit() -> str | None:
    """Current HEAD commit hash, or None outside a git checkout."""
    return _git("rev-parse", "HEAD")


def git_is_dirty() -> bool | None:
    """True when the working tree has uncommitted changes.

    A dirty tree means the recorded commit does not fully describe the code that
    produced the results, so runs flag it explicitly.
    """
    status = _git("status", "--porcelain")
    if status is None:
        return None
    return bool(status)


def collect(seed: int | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Snapshot code + environment provenance for a run.

    Args:
        seed: the run's master random seed, recorded alongside the environment.
        extra: additional key/values to merge in (e.g. config path, dataset).
    """
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": None,
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_name": None,
        "gpu_count": 0,
        "timm_version": None,
        "quantus_version": None,
        "captum_version": None,
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            cudnn = torch.backends.cudnn.version()
            info["cudnn_version"] = str(cudnn) if cudnn else None
    except Exception:  # noqa: BLE001 - provenance must never break a run
        pass

    for mod_name, key in (
        ("timm", "timm_version"),
        ("quantus", "quantus_version"),
        ("captum", "captum_version"),
    ):
        try:
            mod = __import__(mod_name)
            info[key] = getattr(mod, "__version__", None)
        except Exception:  # noqa: BLE001
            pass

    if extra:
        info.update(extra)
    return info


def summary_line(info: dict[str, Any]) -> str:
    """One-line human summary for run logs."""
    commit = (info.get("git_commit") or "unknown")[:8]
    dirty = " (dirty)" if info.get("git_dirty") else ""
    gpu = info.get("gpu_name") or "CPU"
    return (
        f"commit {commit}{dirty} | torch {info.get('torch_version')} | "
        f"cuda {info.get('cuda_version')} | {gpu} | seed {info.get('seed')}"
    )
