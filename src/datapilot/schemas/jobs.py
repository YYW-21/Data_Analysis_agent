from pydantic import BaseModel, Field


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    path: str
    profile: dict


class AnalysisJobRequest(BaseModel):
    dataset_id: str
    user_goal: str = Field(..., min_length=1)
    target_column: str | None = None
    target_columns: list[str] | None = None


class AnalysisJobResponse(BaseModel):
    job_id: str
    status: str
    task_type: str
    target_column: str | None
    best_model: str | None
    report_path: str
    metrics: dict
    artifacts: list[str]
    agent_plan: dict | None = None
    target_columns: list[str] | None = None
    target_results: list[dict] | None = None
