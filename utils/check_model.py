#!/usr/bin/env python3
"""
Scan a directory for JSON files and list those whose `model` field doesn't equal
the target path. When --execute is provided, delete those mismatched files and,
optionally, their corresponding logs (.md and .jsonl) in a provided logs_dir.

Usage examples:
  Dry-run (list only):
    python3 scripts/check_and_delete_mismatched_models.py \
      --results_dir results/webwalkerqa/qwen3_1.7b_sft_grpo_step60

  Delete mismatches and corresponding logs:
    python3 scripts/check_and_delete_mismatched_models.py \
      --results_dir results/webwalkerqa/qwen3_1.7b_sft_grpo_step60 \
      --logs_dir    logs/webwalkerqa/qwen3_1.7b_sft_grpo_step60 \
      --execute
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple



def should_skip_dir(dirname: str) -> bool:
    """Skip hidden/underscore prefixed directories to avoid backups or metadata."""
    base = os.path.basename(dirname)
    return base.startswith(".") or base.startswith("_")


def iter_json_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune unwanted directories in-place
        dirnames[:] = [d for d in dirnames if not should_skip_dir(os.path.join(dirpath, d))]
        for name in filenames:
            if name.endswith(".json"):
                files.append(Path(dirpath) / name)
    return files


def load_model_field(json_path: Path) -> str:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    value = data.get("model")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "List JSON files whose model field mismatches the target; "
            "optionally delete them when --execute is provided."
        )
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default="results/webwalkerqa/qwen3_1.7b_sft_grpo_step300",
        help="Root directory to scan (recursively).",
    )
    parser.add_argument(
        "--logs_dir",
        type=Path,
        default="logs/webwalkerqa/qwen3_1.7b_sft_grpo_step300",
        help="Logs directory to scan (recursively).",
    )
    parser.add_argument(
        "--target_model_path",
        default="/data/group_data/cx_group/verl_agent_shared/checkpoint/apm_1.7b_sft_grpo/global_step_300/huggingface",
        help="Target model path to compare against.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="When set, delete all mismatched JSON files.",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    target_model_path: str = args.target_model_path
    execute: bool = args.execute

    if not results_dir.exists() or not results_dir.is_dir():
        print(f"ERROR: directory does not exist or is not a directory: {results_dir}")
        return 2

    logs_dir: Path = args.logs_dir
    if not logs_dir.exists() or not logs_dir.is_dir():
        print(f"ERROR: logs_dir does not exist or is not a directory: {logs_dir}")
        return 2

    results_json_files = iter_json_files(results_dir)
    total = len(results_json_files)
    mismatches: List[Tuple[Path, str]] = []

    for results_json_file in results_json_files:
        model_value = load_model_field(results_json_file)
        if model_value != target_model_path:
            mismatches.append((results_json_file, model_value))

    # Report
    print(f"Scanned {total} JSON files under: {results_dir}")
    if mismatches:
        print(f"Found {len(mismatches)} mismatched files (TSV: path\tmodel):")
        count = 0
        print(f"First 5 mismatched files:")
        for results_json_file, model_value in mismatches:
            print(f"file: {results_json_file}\nmodel: {model_value}")
            count += 1
            if count >= 5:
                break
    else:
        print("All files match target.")

    if execute and mismatches:
        print("--execute provided: deleting mismatched files...")
        deleted_results = 0
        deleted_logs = 0
        for results_json_file, _ in mismatches:
            try:
                results_json_file.unlink()
                deleted_results += 1
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to delete {results_json_file}: {exc}")

            # Delete corresponding logs 
            file_id = results_json_file.stem
            if file_id.startswith("result_"):
                file_id = file_id[len("result_"):]
            md_file = logs_dir / f"trajectory_{file_id}.md"
            jsonl_file = logs_dir / f"trajectory_{file_id}.jsonl"
            for path in (md_file, jsonl_file):
                if path.exists():
                    try:
                        path.unlink()
                        deleted_logs += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f"Failed to delete {path}: {exc}")
        print(f"Deleted {deleted_results} result files.")
        print(f"Deleted {deleted_logs} log files under: {logs_dir}")

    # Exit codes: 0 if all matched; 1 if mismatches found (even if deleted); 2 for bad args
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())


