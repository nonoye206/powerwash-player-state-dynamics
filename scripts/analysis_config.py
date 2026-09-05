from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(
    explicit_value: str | None,
    env_var: str,
    default_relative: str,
    description: str,
    must_exist: bool = True,
) -> Path:
    """Resolve a data path from CLI value, environment variable, or repo-relative default."""
    if explicit_value:
        path = Path(explicit_value).expanduser()
    elif os.environ.get(env_var):
        path = Path(os.environ[env_var]).expanduser()
    else:
        path = ROOT / default_relative
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}. "
            f"Pass an explicit path or set {env_var}."
        )
    return path


def resolve_raw_zip(explicit_value: str | None = None) -> Path:
    return resolve_path(
        explicit_value,
        env_var="PWS_RAW_ZIP",
        default_relative="data.zip",
        description="Raw telemetry zip archive",
    )


def resolve_raw_data_dir(explicit_value: str | None = None) -> Path:
    return resolve_path(
        explicit_value,
        env_var="PWS_RAW_DATA_DIR",
        default_relative=str(Path("data") / "data"),
        description="Extracted raw telemetry CSV directory",
    )
