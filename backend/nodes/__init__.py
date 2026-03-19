"""Node package scanner -- auto-discovers and loads node packages from backend/nodes/."""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).parent

# Collected from all packages
_node_definitions: dict[str, dict] = {}  # node_type -> yaml definition
_node_executors: dict[str, Callable] = {}  # node_type -> async executor function
_packages: dict[str, dict] = {}  # package_name -> package info


def scan_packages() -> dict[str, dict]:
    """Scan nodes/ directory for packages and load their definitions + executors."""
    _node_definitions.clear()
    _node_executors.clear()
    _packages.clear()

    for pkg_dir in sorted(_PACKAGE_DIR.iterdir()):
        if not pkg_dir.is_dir() or pkg_dir.name.startswith(("_", ".")):
            continue

        node_yaml = pkg_dir / "node.yaml"
        if not node_yaml.exists():
            continue

        try:
            with open(node_yaml) as f:
                pkg_config = yaml.safe_load(f)

            pkg_name = pkg_config.get("name", pkg_dir.name)
            nodes = pkg_config.get("nodes", {})

            # Register node definitions
            for node_type, node_def in nodes.items():
                node_def["_package"] = pkg_name
                _node_definitions[node_type] = node_def

            # Load executor module
            executor_path = pkg_dir / "executor.py"
            if executor_path.exists():
                # Add package dir to sys.path temporarily
                if str(pkg_dir) not in sys.path:
                    sys.path.insert(0, str(pkg_dir))

                spec = importlib.util.spec_from_file_location(
                    f"nodes.{pkg_dir.name}.executor", executor_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Register executors from module's EXECUTORS dict
                executors = getattr(module, "EXECUTORS", {})
                _node_executors.update(executors)

            _packages[pkg_name] = {
                "name": pkg_name,
                "version": pkg_config.get("version", "0.0"),
                "description": pkg_config.get("description", ""),
                "node_count": len(nodes),
                "nodes": list(nodes.keys()),
            }

            logger.info("Loaded node package: %s (%d nodes)", pkg_name, len(nodes))

        except Exception as e:
            logger.warning("Failed to load node package %s: %s", pkg_dir.name, e)

    return _packages


def get_all_definitions() -> dict[str, dict]:
    """Return all registered node definitions."""
    return _node_definitions


def get_all_executors() -> dict[str, Callable]:
    """Return all registered executor functions."""
    return _node_executors


def get_packages() -> dict[str, dict]:
    """Return loaded package info."""
    return _packages
