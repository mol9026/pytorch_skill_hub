"""CLI entry point for unified PyTorch skill discovery and dispatch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .registry import DEFAULT_REGISTRY, register_local_skills


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List or inspect integrated PyTorch skill modules.")
    parser.add_argument("command", choices=["list", "show", "read", "run", "route"], help="Command to run")
    parser.add_argument("name", nargs="?", help="Skill name to inspect or execute")
    parser.add_argument("--root", default=".", help="Project root containing the skills directory")
    return parser


def _skill_dir_for_name(project_root: Path, skill_name: str) -> Path:
    return project_root / "skills" / skill_name


def _detect_entrypoint(skill_dir: Path) -> Path | None:
    candidates = [
        skill_dir / "__main__.py",
        skill_dir / "orchestrator.py",
        skill_dir / "main.py",
        skill_dir / "run.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for search_root in [skill_dir / "scripts", skill_dir / "agent", skill_dir / "src"]:
        if search_root.exists():
            for py_file in sorted(search_root.rglob("*.py")):
                if py_file.name in {"__init__.py"}:
                    continue
                return py_file
    return None


def _route_skill(project_root: Path, text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["pr", "pull request", "review", "code review"]):
        return "pr-review"
    if any(token in lowered for token in ["refactor", "decouple", "device agnostic", "accelerator", "test file"]):
        return "pytorch-test-refactoring"
    if any(token in lowered for token in ["modify", "local update", "merge change", "migration", "hw classification"]):
        return "test-modification-advisor"
    if any(token in lowered for token in ["npu", "diag", "diagnose", "failure", "root cause", "torch_npu"]):
        return "torch-npu-test-failure-diagnose"
    return ""


def main() -> int:
    parser = _build_parser()
    argv = sys.argv[1:]
    args, extra = parser.parse_known_args(argv)

    project_root = Path(args.root).resolve()
    registry = DEFAULT_REGISTRY.__class__()
    register_local_skills(registry, project_root)

    if args.command == "list":
        names = registry.names()
        if not names:
            print("No skills registered.")
            return 0
        for name in names:
            print(name)
        return 0

    if args.command == "show":
        if not args.name:
            parser.error("'show' requires a skill name")
        try:
            spec = registry.get(args.name)
        except KeyError:
            print(f"Skill '{args.name}' was not found.")
            return 1

        print(f"name: {spec.name}")
        print(f"description: {spec.description}")
        print(f"entrypoint: {spec.entrypoint}")
        if spec.tags:
            print(f"tags: {', '.join(spec.tags)}")
        if spec.metadata:
            print("metadata:")
            for key, value in sorted(spec.metadata.items()):
                print(f"  {key}: {value}")
        return 0

    if args.command == "read":
        if not args.name:
            parser.error("'read' requires a skill name")
        try:
            registry.get(args.name)
        except KeyError:
            print(f"Skill '{args.name}' was not found.")
            return 1

        skill_dir = _skill_dir_for_name(project_root, args.name)
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            print(f"Skill '{args.name}' does not contain a SKILL.md file.")
            return 1

        print(skill_file.read_text(encoding="utf-8"))
        return 0

    if args.command == "run":
        if not args.name:
            parser.error("'run' requires a skill name")
        try:
            registry.get(args.name)
        except KeyError:
            print(f"Skill '{args.name}' was not found.")
            return 1

        skill_dir = _skill_dir_for_name(project_root, args.name)
        skill_file = skill_dir / "SKILL.md"
        entrypoint = _detect_entrypoint(skill_dir)

        if entrypoint is None:
            print(f"Skill: {args.name}")
            print(f"Directory: {skill_dir}")
            if skill_file.exists():
                print(f"Skill file: {skill_file}")
            print(f"No executable entrypoint detected for '{args.name}'. This skill is documentation-first; use 'read' to inspect the SKILL.md file.")
            return 0

        passthrough = extra
        print(f"Executing skill '{args.name}' via: {entrypoint}")
        completed = subprocess.run(
            [sys.executable, str(entrypoint), *passthrough],
            cwd=str(skill_dir),
            check=False,
        )
        return completed.returncode

    if args.command == "route":
        if not args.name:
            parser.error("'route' requires a description string")
        matched = _route_skill(project_root, args.name)
        if not matched:
            print("No skill matched the provided description.")
            return 1
        print(matched)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
