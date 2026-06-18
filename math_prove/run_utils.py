"""Shared helpers for safe local runs and prompt experiments."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_INPUT_CANDIDATES = (
    "data/interns1_math_18domains_504.json",
    "sample_data/dev.jsonl",
    "math_prove/validation/core_18_sample.jsonl",
)


def sanitize_run_name(value: str, fallback: str = "run") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:80] or fallback


def find_default_input(candidates: Optional[Iterable[str | Path]] = None) -> Path:
    """Find a reasonable input file when the user does not pass one."""

    env_value = os.environ.get("MATH_PROVE_INPUT") or os.environ.get("CHALLENGE_INPUT_FILE")
    search: list[str | Path] = []
    if env_value:
        search.append(env_value)
    search.extend(candidates or DEFAULT_INPUT_CANDIDATES)

    for candidate in search:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path

    searched = ", ".join(str(Path(c)) for c in search)
    raise FileNotFoundError(
        "No input file was provided and no default dataset was found. "
        f"Searched: {searched}"
    )


def unique_run_dir(root: str | Path, name: str) -> Path:
    """Create a timestamped directory and never reuse an existing one."""

    root_path = Path(root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{stamp}_{sanitize_run_name(name)}"
    candidate = root_path / base_name
    suffix = 2
    while candidate.exists():
        candidate = root_path / f"{base_name}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def non_clobber_path(path: str | Path) -> Path:
    """Return path if free, otherwise add a numeric suffix before the extension."""

    candidate = Path(path)
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        alt = candidate.with_name(f"{candidate.stem}_{suffix}{candidate.suffix}")
        if not alt.exists():
            return alt
        suffix += 1


def plan_batch_paths(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    run_root: str | Path = "outputs/runs",
    run_name: Optional[str] = None,
    results_json_path: Optional[str | Path] = None,
    log_dir: Optional[str | Path] = None,
    summary_path: Optional[str | Path] = None,
    resume: bool = False,
    output_filename: str = "results.jsonl",
) -> dict[str, Path]:
    """Plan a coherent run directory for JSONL, merged JSON, logs, and summary.

    If ``output_path`` is omitted, a new unique run directory is created under
    ``run_root``. If ``output_path`` is provided and already exists, resume may
    append to it; otherwise a non-clobbering sibling path is selected.
    """

    source = Path(input_path)
    name = run_name or source.stem or "run"

    if output_path:
        output = Path(output_path)
        if output.exists() and not resume:
            output = non_clobber_path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        run_dir = output.parent
    else:
        run_dir = unique_run_dir(run_root, name)
        output = run_dir / output_filename

    results_json = Path(results_json_path) if results_json_path else output.with_suffix(".json")
    logs = Path(log_dir) if log_dir else run_dir / "logs"
    summary = Path(summary_path) if summary_path else run_dir / "run_summary.json"

    for path in (results_json.parent, logs, summary.parent):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "output_jsonl": output,
        "results_json": results_json,
        "log_dir": logs,
        "summary": summary,
    }
