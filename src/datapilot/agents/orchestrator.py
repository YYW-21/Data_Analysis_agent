import json
from pathlib import Path
from time import perf_counter

from datapilot.agents.agent_planner import build_agent_plan
from datapilot.agents.report_agent import generate_report
from datapilot.core.storage import job_dir, new_id
from datapilot.schemas.jobs import AnalysisJobResponse
from datapilot.tools.data_loader import load_dataframe
from datapilot.tools.eda import run_eda
from datapilot.tools.ml import train_and_evaluate
from datapilot.tools.profiling import profile_dataframe


def run_analysis_workflow(
    dataset_path: Path,
    user_goal: str,
    target_column: str | None,
) -> AnalysisJobResponse:
    job_id = new_id("job")
    trace = []

    def run_stage(name: str, operation):
        started_at = perf_counter()
        try:
            result = operation()
            trace.append(
                {
                    "stage": name,
                    "status": "completed",
                    "duration_seconds": round(perf_counter() - started_at, 4),
                }
            )
            return result
        except Exception as exc:
            trace.append(
                {
                    "stage": name,
                    "status": "failed",
                    "duration_seconds": round(perf_counter() - started_at, 4),
                    "error": str(exc),
                }
            )
            raise

    df = run_stage("load_dataset", lambda: load_dataframe(dataset_path))
    profile = run_stage("profile_dataset", lambda: profile_dataframe(df))
    task = run_stage(
        "agent_planning",
        lambda: build_agent_plan(
            df, profile=profile, user_goal=user_goal, target_column=target_column
        ),
    )
    artifact_dir = job_dir(job_id, "artifacts")
    artifacts = run_stage(
        "exploratory_analysis",
        lambda: run_eda(df, task["target_column"], artifact_dir),
    )
    ml_result = run_stage(
        "model_training",
        lambda: train_and_evaluate(
            df=df,
            task_type=task["task_type"],
            target_column=task["target_column"],
            model_dir=job_dir(job_id, "models"),
            processed_dir=job_dir(job_id, "processed"),
            artifact_dir=artifact_dir,
        ),
    )
    artifacts.extend(ml_result.get("evaluation_artifacts", []))
    if ml_result.get("feature_importance", {}).get("plot"):
        artifacts.append(ml_result["feature_importance"]["plot"])

    context = {
        "job_id": job_id,
        "dataset_path": str(dataset_path),
        "user_goal": user_goal,
        "profile": profile,
        "task": task,
        "artifacts": artifacts,
        "ml": ml_result,
        "trace": trace,
    }
    report = run_stage("report_generation", lambda: generate_report(context))

    def save_outputs() -> Path:
        report_dir = job_dir(job_id, "reports")
        report_path = report_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        return report_path

    report_path = run_stage("save_outputs", save_outputs)
    context["trace"] = trace
    (report_path.parent / "context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return AnalysisJobResponse(
        job_id=job_id,
        status="completed",
        task_type=task["task_type"],
        target_column=task["target_column"],
        best_model=ml_result["best_model"],
        report_path=str(report_path),
        metrics=ml_result["metrics"],
        artifacts=artifacts,
        agent_plan=task,
        cross_validation=ml_result["cross_validation"],
        model_leaderboard=ml_result["model_leaderboard"],
        trace=trace,
    )
