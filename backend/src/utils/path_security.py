"""Path security utilities."""

from pathlib import Path


def validate_path(target: Path, base_dir: Path) -> Path:
    """Ensure resolved target is under base_dir. Raises ValueError if not."""
    resolved = target.resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ValueError(f"Path traversal blocked: {target}")
    return resolved
