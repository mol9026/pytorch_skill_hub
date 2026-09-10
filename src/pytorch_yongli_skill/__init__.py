"""Public package entry for the PyTorch skill integration scaffold."""

from .registry import (
    DEFAULT_REGISTRY,
    SkillRegistry,
    SkillSpec,
    discover_local_skills,
    register_local_skills,
)

__all__ = [
    "SkillRegistry",
    "SkillSpec",
    "DEFAULT_REGISTRY",
    "discover_local_skills",
    "register_local_skills",
]

__version__ = "0.1.0"
