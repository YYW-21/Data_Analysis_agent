from pathlib import Path

from fastapi import APIRouter, HTTPException

from datapilot.agents.orchestrator import run_analysis_workflow
from datapilot.core.config import settings
from datapilot.schemas.jobs import AnalysisJobRequest, AnalysisJobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=AnalysisJobResponse)
def create_job(request: AnalysisJobRequest) -> AnalysisJobResponse:
    matches = list((settings.data_dir / "datasets").glob(f"{request.dataset_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    try:
        target_columns = _requested_target_columns(request)
        return run_analysis_workflow(
            dataset_path=matches[0],
            user_goal=request.user_goal,
            target_columns=target_columns,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _requested_target_columns(request: AnalysisJobRequest) -> list[str] | None:
    columns: list[str] = []
    if request.target_columns:
        columns.extend(request.target_columns)
    if request.target_column:
        columns.extend(
            part.strip()
            for part in request.target_column.replace("\n", ",").split(",")
            if part.strip()
        )

    unique_columns = list(dict.fromkeys(columns))
    return unique_columns or None


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
