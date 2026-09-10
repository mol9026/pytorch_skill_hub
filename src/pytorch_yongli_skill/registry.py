"""Shared registration utilities for decoupled skill modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class SkillSpec:
    """Metadata describing one skill module."""

    name: str
    description: str = ""
    tags: tuple[str, ...] = ()
    entrypoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _parse_skill_frontmatter(path: Path) -> Optional[SkillSpec]:
    """Parse the YAML front matter from a SKILL.md file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    if not text.startswith("---"):
        return None

    end_index = text.find("\n---", 4)
    if end_index == -1:
        return None

    front_matter = text[4:end_index]
    key_values: Dict[str, str] = {}
    current_key: Optional[str] = None

    for raw_line in front_matter.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            current_key = key
            key_values[key] = value
        elif current_key and line.startswith(" "):
            key_values[current_key] = f"{key_values.get(current_key, '').strip()} {line.strip()}".strip()

    name = (key_values.get("name") or "").strip()
    if not name:
        return None

    description = (key_values.get("description") or "").strip()
    return SkillSpec(
        name=name,
        description=description,
        tags=("local", "skill"),
        entrypoint=str(path),
        metadata={"source_path": str(path)},
    )


def discover_local_skills(root: str | Path) -> List[SkillSpec]:
    """Discover all skill definitions under a project-local skills directory."""
    project_root = Path(root)
    skills_dir = project_root / "skills"
    if not skills_dir.exists():
        return []

    specs: List[SkillSpec] = []
    for candidate_dir in sorted(skills_dir.iterdir()):
        if not candidate_dir.is_dir():
            continue
        skill_file = candidate_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        spec = _parse_skill_frontmatter(skill_file)
        if spec is not None:
            specs.append(spec)
    return specs


def register_local_skills(registry: "SkillRegistry", root: str | Path) -> List[SkillSpec]:
    """Register all discovered local skills in the provided registry."""
    registered: List[SkillSpec] = []
    for spec in discover_local_skills(root):
        registry.register(spec)
        registered.append(spec)
    return registered


class SkillRegistry:
    """Simple in-memory registry for all discovered skills."""

    def __init__(self) -> None:
        self._skills: Dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> SkillSpec:
        self._skills[spec.name] = spec
        return spec

    def register_function(
        self,
        name: str,
        description: str = "",
        tags: Optional[Iterable[str]] = None,
        entrypoint: Optional[str] = None,
        **metadata: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            spec = SkillSpec(
                name=name,
                description=description,
                tags=tuple(tags or ()),
                entrypoint=entrypoint or getattr(func, "__qualname__", None),
                metadata=metadata,
            )
            self.register(spec)
            return func

        return decorator

    def get(self, name: str) -> SkillSpec:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"Skill '{name}' is not registered.") from exc

    def list(self) -> List[SkillSpec]:
        return list(self._skills.values())

    def names(self) -> List[str]:
        return list(self._skills.keys())

    def clear(self) -> None:
        self._skills.clear()


DEFAULT_REGISTRY = SkillRegistry()
try:
    register_local_skills(DEFAULT_REGISTRY, Path(__file__).resolve().parents[2])
except Exception:
    pass
