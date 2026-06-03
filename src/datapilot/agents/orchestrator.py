import json
from pathlib import Path

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
    df = load_dataframe(dataset_path)

    profile = profile_dataframe(df)
    task = build_agent_plan(df, profile=profile, user_goal=user_goal, target_column=target_column)
    artifacts = run_eda(df, task["target_column"], job_dir(job_id, "artifacts"))
    ml_result = train_and_evaluate(
        df=df,
        task_type=task["task_type"],
        target_column=task["target_column"],
        model_dir=job_dir(job_id, "models"),
    )

    context = {
        "job_id": job_id,
        "dataset_path": str(dataset_path),
        "user_goal": user_goal,
        "profile": profile,
        "task": task,
        "artifacts": artifacts,
        "ml": ml_result,
    }
    report = generate_report(context)

    report_dir = job_dir(job_id, "reports")
    report_path = report_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    (report_dir / "context.json").write_text(
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
    )
