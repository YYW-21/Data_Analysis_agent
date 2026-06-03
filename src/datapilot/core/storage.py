from pathlib import Path
from uuid import uuid4

from datapilot.core.config import settings


def ensure_storage_dirs() -> None:
    for name in ["datasets", "processed", "artifacts", "models", "reports"]:
        (settings.data_dir / name).mkdir(parents=True, exist_ok=True)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def dataset_path(dataset_id: str, suffix: str) -> Path:
    return settings.data_dir / "datasets" / f"{dataset_id}{suffix}"


def job_dir(job_id: str, folder: str) -> Path:
    path = settings.data_dir / folder / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path

