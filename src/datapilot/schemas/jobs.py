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
    cross_validation: dict | None = None
    model_leaderboard: list[dict] | None = None
    trace: list[dict] | None = None


class PredictionResponse(BaseModel):
    job_id: str
    prediction_path: str
    row_count: int
    preview: list[dict]
