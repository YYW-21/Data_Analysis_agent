import json
import re
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
    target_columns: list[str] | None = None,
    target_column: str | None = None,
) -> AnalysisJobResponse:
    job_id = new_id("job")
    df = load_dataframe(dataset_path)

    profile = profile_dataframe(df)
    requested_targets = _normalize_target_columns(target_columns, target_column)
    if requested_targets and len(requested_targets) > 1:
        return _run_multi_target_workflow(
            job_id=job_id,
            df=df,
            dataset_path=dataset_path,
            profile=profile,
            user_goal=user_goal,
            target_columns=requested_targets,
        )

    selected_target = requested_targets[0] if requested_targets else None
    task = build_agent_plan(df, profile=profile, user_goal=user_goal, target_column=selected_target)
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
        target_columns=[task["target_column"]],
        target_results=None,
    )


def _run_multi_target_workflow(
    job_id: str,
    df,
    dataset_path: Path,
    profile: dict,
    user_goal: str,
    target_columns: list[str],
) -> AnalysisJobResponse:
    results: list[dict] = []
    all_artifacts: list[str] = []

    for target in target_columns:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' does not exist.")

        task = build_agent_plan(df, profile=profile, user_goal=user_goal, target_column=target)
        safe_target = _safe_name(task["target_column"])
        artifact_dir = job_dir(job_id, "artifacts") / safe_target
        model_dir = job_dir(job_id, "models") / safe_target

        artifacts = run_eda(df, task["target_column"], artifact_dir)
        ml_result = train_and_evaluate(
            df=df,
            task_type=task["task_type"],
            target_column=task["target_column"],
            model_dir=model_dir,
            exclude_columns=[col for col in target_columns if col != task["target_column"]],
        )
        all_artifacts.extend(artifacts)
        results.append(
            {
                "target_column": task["target_column"],
                "task_type": task["task_type"],
                "best_model": ml_result["best_model"],
                "metrics": ml_result["metrics"],
                "artifacts": artifacts,
                "agent_plan": task,
                "ml": ml_result,
            }
        )

    context = {
        "job_id": job_id,
        "dataset_path": str(dataset_path),
        "user_goal": user_goal,
        "profile": profile,
        "multi_target": True,
        "target_columns": [result["target_column"] for result in results],
        "target_results": results,
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
        task_type="multi_target",
        target_column=None,
        best_model=None,
        report_path=str(report_path),
        metrics={result["target_column"]: result["metrics"] for result in results},
        artifacts=all_artifacts,
        agent_plan={
            "source": "multi_target",
            "target_count": len(results),
            "targets": [result["agent_plan"] for result in results],
        },
        target_columns=[result["target_column"] for result in results],
        target_results=[
            {
                "target_column": result["target_column"],
                "task_type": result["task_type"],
                "best_model": result["best_model"],
                "metrics": result["metrics"],
                "agent_plan": result["agent_plan"],
                "artifacts": result["artifacts"],
            }
            for result in results
        ],
    )


def _normalize_target_columns(
    target_columns: list[str] | None,
    target_column: str | None,
) -> list[str]:
    values: list[str] = []
    if target_columns:
        values.extend(target_columns)
    if target_column:
        values.extend(
            part.strip()
            for part in target_column.replace("\n", ",").split(",")
            if part.strip()
        )
    return list(dict.fromkeys(values))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "target"
