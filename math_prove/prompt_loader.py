"""Utility to load system prompt overrides from variant directories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

PROMPT_FILES: Dict[str, str] = {
    "classify": "classify.txt",
    "solve":    "solve.txt",
    "verify":   "verify.txt",
    "select":   "select.txt",
    "extract":  "extract.txt",
}


def load_prompt_overrides(variant: str, base_dir: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Scan <base_dir>/<variant>/ for the 5 fixed .txt files.

    Empty/missing entries become None. Raises FileNotFoundError if the
    variant directory is missing entirely. Rejects variants with '/' or '\\'
    to prevent path traversal.
    """
    if (
        "/" in variant
        or "\\" in variant
        or variant.strip() in {"", ".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", variant.strip())
    ):
        raise ValueError(f"Invalid variant name: {variant}")

    if base_dir is None:
        # Default to repo_root/prompts
        base_dir_path = Path(__file__).parent.parent / "prompts"
    else:
        base_dir_path = Path(base_dir)

    variant_dir = base_dir_path / variant
    if not variant_dir.exists():
        raise FileNotFoundError(
            f"Variant directory not found: {variant_dir.absolute()}.\n"
            f"Expected directory containing one or more of: {list(PROMPT_FILES.values())}"
        )

    overrides: Dict[str, Optional[str]] = {}
    for stage, filename in PROMPT_FILES.items():
        file_path = variant_dir / filename
        if file_path.is_file():
            content = file_path.read_text(encoding="utf-8").strip()
            overrides[stage] = content if content else None
        else:
            overrides[stage] = None

    return overrides


def fallback_stage_count(overrides: Dict[str, Optional[str]]) -> int:
    """Returns how many stages fall back to the default prompts."""
    return sum(1 for v in overrides.values() if v is None)
