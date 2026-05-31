"""Path security utilities."""

from pathlib import Path


def validate_path(target: Path, base_dir: Path) -> Path:
    """Ensure resolved target is under base_dir. Raises ValueError if not.

    round7:用 is_relative_to 而非 str(...).startswith —— 前缀匹配有经典缺陷,
    base=`/x/agents` 会放行 `/x/agents-evil/...`(同前缀兄弟目录逃逸)。
    """
    resolved = target.resolve()
    base = base_dir.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"Path traversal blocked: {target}")
    return resolved
