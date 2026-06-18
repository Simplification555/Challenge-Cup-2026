import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from llm_client import InternChatClient
from math_prove.run_utils import find_default_input, unique_run_dir
from user_agent import ReasoningAgent


LOCAL_MAX_CONCURRENCY = int(os.environ.get("LOCAL_MAX_CONCURRENCY", "8"))


def load_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if not line.strip():
                continue
            item = json.loads(line)
            item.setdefault("idx", line_number)
            items.append(item)
    return items


def result_path(output_dir: Path, item: Dict) -> Path:
    return output_dir / f"{item['idx']}.json"


def is_processed(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def write_json(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
        file.write("\n")
    tmp_path.replace(path)


def build_output_record(item: Dict, agent_result: Dict) -> Dict:
    final_response = agent_result.get("final_response", "")
    if not isinstance(final_response, str) or not final_response.strip():
        raise ValueError("agent.solve must return a non-empty string field: final_response")

    output = {
        "idx": item["idx"],
        "status": "success",
        "final_response": final_response,
        "trace": agent_result.get("trace", []),
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Competition sample reasoning agent.")
    parser.add_argument(
        "--input_file",
        default=None,
        help=(
            "Path to input JSONL. If omitted, uses CHALLENGE_INPUT_FILE, "
            "MATH_PROVE_INPUT, sample_data/dev.jsonl, or another known local dataset."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for per-problem JSON outputs. If omitted, a unique directory is created.",
    )
    parser.add_argument("--run_name", default=None, help="Optional name used in auto output dirs.")
    return parser.parse_args()


def solve_item(agent: ReasoningAgent, item: Dict) -> Dict:
    result = agent.solve(
        problem=item["problem"],
        metadata={"idx": item["idx"]},
    )
    return build_output_record(item, result)


async def process_item(
    agent: ReasoningAgent,
    item: Dict,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    path = result_path(output_dir, item)
    if is_processed(path):
        print(f"Skip idx={item['idx']} because {path} already exists.")
        return

    async with semaphore:
        try:
            record = await asyncio.to_thread(solve_item, agent, item)
        except Exception as exc:  # noqa: BLE001 - keep one output file per input item.
            record = {
                "idx": item["idx"],
                "status": "error",
                "final_response": "",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "trace": [],
            }
        await asyncio.to_thread(write_json, path, record)
        print(f"Finished idx={item['idx']}")


async def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input_file) if args.input_file else find_default_input(
        ("sample_data/dev.jsonl", "data/interns1_math_18domains_504.json")
    )
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = unique_run_dir("outputs/official_runs", args.run_name or input_path.stem)

    items = load_jsonl(input_path)

    client = InternChatClient()
    agent = ReasoningAgent(client=client)
    semaphore = asyncio.Semaphore(LOCAL_MAX_CONCURRENCY)
    start = time.time()

    print(f"Loaded {len(items)} items. Max concurrency: {LOCAL_MAX_CONCURRENCY}.")
    print(f"Input: {input_path}")
    print(f"Output dir: {output_dir}")
    tasks = [process_item(agent, item, output_dir, semaphore) for item in items]
    await asyncio.gather(*tasks)
    summary = {
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "total_loaded": len(items),
        "max_concurrency": LOCAL_MAX_CONCURRENCY,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    write_json(output_dir / "run_summary.json", summary)
    print(f"Saved outputs to {output_dir}")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
