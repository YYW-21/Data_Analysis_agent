from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from datapilot.agents.orchestrator import run_analysis_workflow
from datapilot.core.config import settings
from datapilot.core.storage import job_dir, new_id
from datapilot.schemas.jobs import AnalysisJobRequest, AnalysisJobResponse, PredictionResponse
from datapilot.tools.data_loader import load_dataframe
from datapilot.tools.ml import predict_with_model

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=AnalysisJobResponse)
def create_job(request: AnalysisJobRequest) -> AnalysisJobResponse:
    matches = list((settings.data_dir / "datasets").glob(f"{request.dataset_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    try:
        return run_analysis_workflow(
            dataset_path=matches[0],
            user_goal=request.user_goal,
            target_column=request.target_column,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{job_id}/report")
def get_report(job_id: str) -> dict[str, str]:
    report_path = settings.data_dir / "reports" / job_id / "report.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return {"job_id": job_id, "markdown": report_path.read_text(encoding="utf-8")}


@router.get("/{job_id}/artifacts")
def get_artifacts(job_id: str) -> dict[str, list[str]]:
    artifact_dir = settings.data_dir / "artifacts" / job_id
    if not artifact_dir.exists():
        raise HTTPException(status_code=404, detail="Artifacts not found.")
    return {"job_id": job_id, "artifacts": [str(path) for path in Path(artifact_dir).glob("*")]}


@router.post("/{job_id}/predict", response_model=PredictionResponse)
async def predict_job(
    job_id: str,
    file: Annotated[UploadFile, File(...)],
) -> PredictionResponse:
    context_path = settings.data_dir / "reports" / job_id / "context.json"
    if not context_path.exists():
        raise HTTPException(status_code=404, detail="Job context not found.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported.")

    import json

    context = json.loads(context_path.read_text(encoding="utf-8"))
    model_path = Path(context["ml"]["model_path"])
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found.")

    temp_path = job_dir(job_id, "processed") / f"predict_input_{new_id('pred')}{suffix}"
    temp_path.write_bytes(await file.read())

    try:
        df = load_dataframe(temp_path)
        predictions = predict_with_model(model_path, df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    output_path = job_dir(job_id, "processed") / f"predictions_{new_id('pred')}.csv"
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    return PredictionResponse(
        job_id=job_id,
        prediction_path=str(output_path),
        row_count=len(predictions),
        preview=predictions.head(20).where(predictions.notna(), None).to_dict(orient="records"),
    )

