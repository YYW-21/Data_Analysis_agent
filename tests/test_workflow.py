from pathlib import Path

import pandas as pd
import pytest

from datapilot.agents.agent_planner import build_agent_plan
from datapilot.agents.orchestrator import run_analysis_workflow
from datapilot.tools.ml import predict_with_model, train_and_evaluate
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


def test_agent_plan_falls_back_to_rules_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("datapilot.agents.agent_planner.settings.enable_agent_workflow", True)
    monkeypatch.setattr("datapilot.agents.agent_planner.settings.openai_api_key", "")
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "churn": [0, 1, 0, 1],
        }
    )
    profile = profile_dataframe(df)

    plan = build_agent_plan(df, profile=profile, user_goal="predict churn", target_column=None)

    assert plan["source"] == "rules"
    assert plan["target_column"] == "churn"
    assert plan["task_type"] == "classification"


def test_run_analysis_workflow_on_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("datapilot.agents.agent_planner.settings.enable_agent_workflow", False)
    monkeypatch.setattr("datapilot.agents.report_agent.settings.enable_llm_report", False)
    result = run_analysis_workflow(
        dataset_path=Path("examples/titanic_sample.csv"),
        user_goal="predict survival",
        target_column="survived",
    )

    assert result.status == "completed"
    assert result.task_type == "classification"
    assert result.best_model is not None
    assert Path(result.report_path).exists()
    assert result.model_leaderboard
    assert result.cross_validation
    assert result.trace
    assert {item["stage"] for item in result.trace} >= {
        "load_dataset",
        "agent_planning",
        "model_training",
        "report_generation",
    }


def test_training_uses_advanced_candidate_models(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "age": [22, 38, 26, 35, 54, 2, 27, 14, 58, 20, 39, 31],
            "fare": [7, 71, 8, 53, 52, 21, 11, 30, 27, 8, 31, 18],
            "sex": [
                "male",
                "female",
                "female",
                "female",
                "male",
                "male",
                "female",
                "female",
                "female",
                "male",
                "male",
                "female",
            ],
            "survived": [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
        }
    )

    artifact_dir = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"
    result = train_and_evaluate(
        df,
        "classification",
        "survived",
        tmp_path / "models",
        processed_dir=processed_dir,
        artifact_dir=artifact_dir,
    )
    trained_models = {item["model"] for item in result["candidate_metrics"]}

    assert {"extra_trees", "gradient_boosting", "hist_gradient_boosting", "xgboost"}.issubset(
        trained_models
    )
    assert Path(result["processed"]["cleaned_csv"]).exists()
    assert Path(result["processed"]["preprocessing_summary_json"]).exists()
    cleaned = pd.read_csv(result["processed"]["cleaned_csv"])
    feature_preview = pd.read_csv(result["processed"]["feature_preview_csv"])
    assert int(cleaned.isna().sum().sum()) == 0
    assert int(feature_preview.isna().sum().sum()) == 0
    assert result["evaluation_artifacts"]
    assert result["model_leaderboard"]
    assert result["model_leaderboard"][0]["is_best"]
    assert result["cross_validation"]["folds"] >= 2
    assert all("cross_validation" in item for item in result["candidate_metrics"])
    assert Path(result["evaluation_artifacts"][0]).exists()
    if result["feature_importance"]:
        assert Path(result["feature_importance"]["plot"]).exists()

    predictions = predict_with_model(Path(result["model_path"]), df.drop(columns=["survived"]))
    assert "prediction" in predictions.columns
    assert len(predictions) == len(df)
