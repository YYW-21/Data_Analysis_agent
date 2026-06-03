import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine

from datapilot.agents.agent_planner import build_agent_plan
from datapilot.core.config import settings
from datapilot.core.storage import new_id
from datapilot.tools.ml import train_and_evaluate
from datapilot.tools.profiling import profile_dataframe


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    data: pd.DataFrame
    user_goal: str
    expected_target: str
    expected_task_type: str


def benchmark_cases() -> list[BenchmarkCase]:
    return [
        _sklearn_case(load_breast_cancer, "breast_cancer", "predict diagnosis", "classification"),
        _sklearn_case(load_iris, "iris", "predict flower species", "classification"),
        _sklearn_case(load_wine, "wine", "predict wine class", "classification"),
        _sklearn_case(load_diabetes, "diabetes", "predict disease progression", "regression"),
        _titanic_case(),
    ]


def run_benchmark(
    output_dir: Path | None = None,
    use_agent: bool = False,
    max_cases: int | None = None,
) -> dict:
    previous_agent_setting = settings.enable_agent_workflow
    settings.enable_agent_workflow = use_agent

    run_id = new_id("bench")
    output_dir = output_dir or Path("benchmarks") / "results" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = benchmark_cases()
    if max_cases is not None:
        selected_cases = selected_cases[:max_cases]

    started_at = time.time()
    results = []
    try:
        for case in selected_cases:
            results.append(_run_case(case, output_dir))
    finally:
        settings.enable_agent_workflow = previous_agent_setting

    summary = _summarize(results, started_at, use_agent)
    payload = {
        "run_id": run_id,
        "summary": summary,
        "results": results,
    }
    _write_outputs(payload, output_dir)
    return payload


def _run_case(case: BenchmarkCase, output_dir: Path) -> dict:
    case_dir = output_dir / case.name
    model_dir = case_dir / "models"
    processed_dir = case_dir / "processed"
    artifact_dir = case_dir / "artifacts"
    case_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    profile = profile_dataframe(case.data)
    result = {
        "dataset": case.name,
        "rows": len(case.data),
        "columns": len(case.data.columns),
        "expected_target": case.expected_target,
        "expected_task_type": case.expected_task_type,
        "status": "failed",
        "error": None,
    }

    try:
        plan = build_agent_plan(
            case.data,
            profile=profile,
            user_goal=case.user_goal,
            target_column=None,
        )
        target_correct = plan["target_column"] == case.expected_target
        task_type_correct = plan["task_type"] == case.expected_task_type
        train_result = train_and_evaluate(
            df=case.data,
            task_type=plan["task_type"],
            target_column=plan["target_column"],
            model_dir=model_dir,
            processed_dir=processed_dir,
            artifact_dir=artifact_dir,
        )
        result.update(
            {
                "status": "passed",
                "inferred_target": plan["target_column"],
                "inferred_task_type": plan["task_type"],
                "target_correct": target_correct,
                "task_type_correct": task_type_correct,
                "agent_source": plan.get("source", "rules"),
                "best_model": train_result["best_model"],
                "metrics": train_result["metrics"],
                "failed_models": train_result.get("failed_models", []),
                "duration_seconds": round(time.time() - started_at, 3),
            }
        )
    except Exception as exc:
        result.update(
            {
                "error": str(exc),
                "duration_seconds": round(time.time() - started_at, 3),
            }
        )
    return result


def _summarize(results: list[dict], started_at: float, use_agent: bool) -> dict:
    total = len(results)
    passed = sum(1 for item in results if item["status"] == "passed")
    target_correct = sum(1 for item in results if item.get("target_correct"))
    task_type_correct = sum(1 for item in results if item.get("task_type_correct"))
    agent_used = sum(1 for item in results if item.get("agent_source") == "agent")
    rule_used = sum(1 for item in results if item.get("agent_source") == "rules")
    return {
        "total_cases": total,
        "passed_cases": passed,
        "pipeline_success_rate": _rate(passed, total),
        "target_inference_accuracy": _rate(target_correct, total),
        "task_type_accuracy": _rate(task_type_correct, total),
        "agent_enabled": use_agent,
        "agent_used_cases": agent_used,
        "rule_used_cases": rule_used,
        "total_duration_seconds": round(time.time() - started_at, 3),
    }


def _write_outputs(payload: dict, output_dir: Path) -> None:
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(payload["results"], output_dir / "benchmark_results.csv")
    _write_markdown(payload, output_dir / "benchmark_report.md")


def _write_csv(results: list[dict], path: Path) -> None:
    fields = [
        "dataset",
        "status",
        "expected_target",
        "inferred_target",
        "target_correct",
        "expected_task_type",
        "inferred_task_type",
        "task_type_correct",
        "agent_source",
        "best_model",
        "duration_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({field: item.get(field) for field in fields})


def _write_markdown(payload: dict, path: Path) -> None:
    summary = payload["summary"]
    lines = [
        "# DataPilot Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Total cases: {summary['total_cases']}",
        f"- Passed cases: {summary['passed_cases']}",
        f"- Pipeline success rate: {summary['pipeline_success_rate']:.2%}",
        f"- Target inference accuracy: {summary['target_inference_accuracy']:.2%}",
        f"- Task type accuracy: {summary['task_type_accuracy']:.2%}",
        f"- Agent enabled: {summary['agent_enabled']}",
        f"- Agent used cases: {summary['agent_used_cases']}",
        f"- Rule used cases: {summary['rule_used_cases']}",
        f"- Total duration seconds: {summary['total_duration_seconds']}",
        "",
        "## Cases",
        "",
        "| Dataset | Status | Target | Task | Best model | Duration |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for item in payload["results"]:
        lines.append(
            "| "
            f"{item['dataset']} | "
            f"{item['status']} | "
            f"{item.get('inferred_target', '-')} | "
            f"{item.get('inferred_task_type', '-')} | "
            f"{item.get('best_model', '-')} | "
            f"{item.get('duration_seconds', '-')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sklearn_case(loader, name: str, goal: str, task_type: str) -> BenchmarkCase:
    bunch = loader(as_frame=True)
    data = bunch.frame.copy()
    target_col = "target"
    if target_col not in data.columns:
        data[target_col] = bunch.target
    return BenchmarkCase(
        name=name,
        data=data,
        user_goal=goal,
        expected_target=target_col,
        expected_task_type=task_type,
    )


def _titanic_case() -> BenchmarkCase:
    data = pd.read_csv("examples/titanic_sample.csv")
    return BenchmarkCase(
        name="titanic_sample",
        data=data,
        user_goal="predict passenger survival",
        expected_target="survived",
        expected_task_type="classification",
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DataPilot benchmark suite.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--use-agent", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    payload = run_benchmark(
        output_dir=args.output_dir,
        use_agent=args.use_agent,
        max_cases=args.max_cases,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
