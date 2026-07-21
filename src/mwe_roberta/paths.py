from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the root of the installed source checkout."""
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    override = os.environ.get("PARSEME_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return project_root().parent / "sharedtask-data" / "2.0" / "subtask1"


def default_artifacts_dir() -> Path:
    override = os.environ.get("MWE_ARTIFACTS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "artifacts"
