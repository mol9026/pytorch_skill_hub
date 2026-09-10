#!/usr/bin/env python3
"""Compare before/after collected test ID manifests.

Inputs may be newline-delimited text, a JSON list, or a JSON object containing
`test_ids`. The tool is deterministic and never starts a test runner.
"""
import argparse
import json
from pathlib import Path


def load_manifest(path: str) -> set[str]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")}
    if isinstance(data, dict):
        data = data.get("test_ids")
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"{path}: expected a JSON string list or an object with test_ids")
    return set(data)


def compare(before: set[str], after: set[str]) -> dict:
    return {
        "before_count": len(before),
        "after_count": len(after),
        "added": sorted(after - before),
        "removed": sorted(before - after),
        "unchanged": sorted(before & after),
    }


def render(result: dict) -> str:
    lines = [
        "# Collection manifest diff",
        f"before {result['before_count']} | after {result['after_count']} | "
        f"added {len(result['added'])} | removed {len(result['removed'])} | unchanged {len(result['unchanged'])}",
        "",
        "## Added",
        *([f"- `{item}`" for item in result["added"]] or ["（无）"]),
        "",
        "## Removed",
        *([f"- `{item}`" for item in result["removed"]] or ["（无）"]),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare collected test ID manifests without running tests.")
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = compare(load_manifest(args.before), load_manifest(args.after))
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else render(result))


if __name__ == "__main__":
    main()
