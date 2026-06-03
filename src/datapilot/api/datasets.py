from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from datapilot.core.config import settings
from datapilot.core.storage import dataset_path, new_id
from datapilot.schemas.jobs import DatasetUploadResponse
from datapilot.tools.data_loader import load_dataframe
from datapilot.tools.profiling import profile_dataframe

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: Annotated[UploadFile, File(...)]) -> DatasetUploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported.")

    dataset_id = new_id("ds")
    path = dataset_path(dataset_id, suffix)
    path.write_bytes(await file.read())

    try:
        df = load_dataframe(path)
        profile = profile_dataframe(df)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to read dataset: {exc}") from exc

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        filename=file.filename or path.name,
        path=str(path),
        profile=profile,
    )


@router.get("/{dataset_id}/profile")
def get_dataset_profile(dataset_id: str) -> dict:
    matches = list((settings.data_dir / "datasets").glob(f"{dataset_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    df = load_dataframe(matches[0])
    return profile_dataframe(df)
