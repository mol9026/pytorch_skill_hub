"""Skill modules package.

This is the integration surface for decoupled PyTorch use-case modules.
Each module can register itself here or via the shared `DEFAULT_REGISTRY`.
"""

from ..registry import DEFAULT_REGISTRY, SkillRegistry

__all__ = ["DEFAULT_REGISTRY", "SkillRegistry"]
