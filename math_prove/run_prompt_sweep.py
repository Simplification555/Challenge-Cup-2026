"""Run batch reasoning sweeps over multiple prompt variant directories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from .main import run_batch, _resolve_api_config
from .prompt_loader import load_prompt_overrides, fallback_stage_count
from .run_utils import find_default_input, sanitize_run_name, unique_run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run prompt sweeps over variant directories (solve/classify/verify/select/extract)."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="Input dataset path. If omitted, searches default locations.",
    )
    parser.add_argument(
        "--variants",
        required=True,
        help="Comma-separated variant subdirectory names under --prompts-dir (e.g. v1_baseline,v2_strict).",
    )
    parser.add_argument(
        "--prompts-dir",
        default="prompts",
        help="Base directory containing variant folders. Default is 'prompts'.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Base output directory for the sweep. If omitted, a unique directory "
            "under outputs/prompt_sweeps is created."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional name used when --output-dir is omitted.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="intern-s2-preview",
        help="Model type name (default: intern-s2-preview).",
    )
    parser.add_argument("--api-key", default=None, help="API key override.")
    parser.add_argument("--api-base", default=None, help="API base URL override.")
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Only process the first N problems.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each variant run by skipping already processed problems.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON/YAML config file.",
    )
    parser.add_argument(
        "--ablation",
        default="full",
        help="Ablation preset (default: full).",
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="Run in official mode (validates keys and endpoints).",
    )
    # Reserved but currently unused flags
    parser.add_argument("--workers", type=int, default=1, help="Reserved for future parallel execution.")
    parser.add_argument("--rpm-limit", type=int, default=None, help="Reserved for future rate limiting.")
    return parser


def run_sweep(args: argparse.Namespace) -> int:
    input_path = Path(args.input) if args.input else find_default_input()
    if args.output_dir:
        output_root = Path(args.output_dir)
        if output_root.exists() and any(output_root.iterdir()) and not args.resume:
            output_root = unique_run_dir(output_root.parent, output_root.name)
        else:
            output_root.mkdir(parents=True, exist_ok=True)
    else:
        output_root = unique_run_dir(
            "outputs/prompt_sweeps",
            args.run_name or f"prompt_sweep_{input_path.stem}",
        )

    # Resolve variants list
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not variants:
        print("Error: No variants specified in --variants.")
        return 1

    api_key, api_base = _resolve_api_config(args)

    print("=" * 72)
    print("PROMPT SWEEP RUNNER")
    print(f"Input path:   {input_path}")
    print(f"Output root:  {output_root}")
    print(f"Model:        {args.model}")
    print(f"Ablation:     {args.ablation}")
    print(f"Variants:     {', '.join(variants)}")
    print("=" * 72)

    sweep_results: List[Dict[str, Any]] = []
    any_failed = False

    for index, variant in enumerate(variants, start=1):
        variant_dir_name = sanitize_run_name(variant)
        variant_output_dir = _plan_variant_output_dir(output_root, variant_dir_name, resume=args.resume)

        print("\n" + "#" * 72)
        print(f"[{index}/{len(variants)}] Running variant: {variant}")
        print(f"Output directory: {variant_output_dir}")

        # 1. Load prompt overrides
        try:
            overrides = load_prompt_overrides(variant, base_dir=args.prompts_dir)
            fallbacks = fallback_stage_count(overrides)
            loaded_stages = len(overrides) - fallbacks
            print(f"Loaded overrides for {loaded_stages}/5 stages (fallback default={fallbacks})")
        except Exception as exc:
            print(f"ERROR: Failed to load overrides for variant '{variant}': {exc}")
            sweep_results.append({
                "variant": variant,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": 0.0,
            })
            any_failed = True
            continue

        # 2. Execute the run_batch
        start_time = time.time()
        try:
            variant_output_dir.mkdir(parents=True, exist_ok=True)
            summary = run_batch(
                input_path=str(input_path),
                output_path=str(variant_output_dir / "results.jsonl"),
                model_type=args.model,
                api_key=api_key,
                api_base=api_base,
                limit=args.limit,
                resume=args.resume,
                results_json_path=str(variant_output_dir / "results.json"),
                log_dir=str(variant_output_dir / "logs"),
                summary_path=str(variant_output_dir / "run_summary.json"),
                config_path=args.config,
                ablation=args.ablation,
                official_mode=args.official,
                prompt_overrides=overrides,
            )
            elapsed = time.time() - start_time
            print(f"Variant '{variant}' completed successfully in {elapsed:.1f}s.")
            sweep_results.append({
                "variant": variant,
                "status": "success",
                "results_in_jsonl": summary.get("results_in_jsonl", 0),
                "fallback_stages": fallbacks,
                "elapsed_seconds": round(elapsed, 3),
            })
        except Exception as exc:
            elapsed = time.time() - start_time
            print(f"ERROR: Variant '{variant}' failed after {elapsed:.1f}s: {exc}")
            sweep_results.append({
                "variant": variant,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(elapsed, 3),
            })
            any_failed = True

    # Write overall sweep_summary.json
    sweep_summary_path = output_root / "sweep_summary.json"
    sweep_summary = {
        "input_path": str(input_path),
        "output_dir": str(output_root),
        "model": args.model,
        "ablation": args.ablation,
        "variants": sweep_results,
    }
    sweep_summary_path.write_text(json.dumps(sweep_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print final summary table
    print("\n" + "=" * 72)
    print("SWEEP SUMMARY REPORT")
    print("-" * 72)
    print(f"{'Variant':<20} | {'Status':<8} | {'Results':<8} | {'Fallback':<8} | {'Time (s)':<10}")
    print("-" * 72)
    for res in sweep_results:
        var = res["variant"][:20]
        status = res["status"]
        count = res.get("results_in_jsonl", "-")
        fallbacks = res.get("fallback_stages", "-")
        secs = f"{res['elapsed_seconds']:.1f}"
        print(f"{var:<20} | {status:<8} | {count:<8} | {fallbacks:<8} | {secs:<10}")
    print("=" * 72)
    print(f"Summary saved to: {sweep_summary_path}")

    return 1 if any_failed else 0


def _plan_variant_output_dir(output_root: Path, variant_name: str, resume: bool) -> Path:
    """Choose a per-variant directory without overwriting previous runs."""

    base = output_root / variant_name
    if resume or not base.exists():
        return base
    if not any(base.iterdir()):
        return base
    suffix = 2
    while True:
        candidate = output_root / f"{variant_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def main() -> None:
    parser = build_arg_parser()
    sys.exit(run_sweep(parser.parse_args()))


if __name__ == "__main__":
    main()
