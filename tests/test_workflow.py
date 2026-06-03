from pathlib import Path

import pandas as pd

from datapilot.agents.orchestrator import run_analysis_workflow
from datapilot.tools.profiling import profile_dataframe
from datapilot.tools.task_inference import infer_task


def test_profile_dataframe_detects_basic_shape() -> None:
    df = pd.DataFrame(
        {
            "age": [20, 30, None],
            "city": ["A", "B", "A"],
            "churn": [0, 1, 0],
        }
    )

    profile = profile_dataframe(df)

    assert profile["rows"] == 3
    assert profile["columns"] == 3
    assert profile["missing_values"]["age"] == 1
    assert "churn" in profile["target_candidates"]


def test_infer_task_classification() -> None:
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "churn": [0, 1, 0, 1],
        }
    )

    task = infer_task(df, user_goal="predict churn", target_column=None)

    assert task["task_type"] == "classification"
    assert task["target_column"] == "churn"


def test_run_analysis_workflow_on_sample() -> None:
    result = run_analysis_workflow(
        dataset_path=Path("examples/titanic_sample.csv"),
        user_goal="predict survival",
        target_column="survived",
    )

    assert result.status == "completed"
    assert result.task_type == "classification"
    assert result.best_model is not None
    assert Path(result.report_path).exists()
