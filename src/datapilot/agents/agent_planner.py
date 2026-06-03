import json

import pandas as pd
from agents import Agent, OpenAIProvider, RunConfig, Runner
from pydantic import BaseModel, Field

from datapilot.core.config import settings
from datapilot.tools.task_inference import infer_task


class AgentWorkflowPlan(BaseModel):
    target_column: str | None = Field(
        default=None,
        description="The dataset column to predict or analyze as target. Must match a real column.",
    )
    task_type: str = Field(
        description="One of: classification, regression, eda_only.",
    )
    metric: str = Field(description="Recommended primary metric, such as f1_macro or r2.")
    analysis_focus: list[str] = Field(
        default_factory=list,
        description="Key analysis points the user likely cares about.",
    )
    workflow_steps: list[str] = Field(
        default_factory=list,
        description="Short ordered steps for this analysis workflow.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


def build_agent_plan(
    df: pd.DataFrame,
    profile: dict,
    user_goal: str,
    target_column: str | None,
) -> dict:
    fallback = _try_rule_fallback(df, user_goal, target_column)

    if not settings.enable_agent_workflow:
        if fallback is None:
            raise ValueError("Could not infer target column. Please provide target_column.")
        return _fallback_plan(fallback, "Rule-based fallback plan.")
    if not _has_real_api_key():
        if fallback is None:
            raise ValueError(
                "Agent workflow is enabled, but OPENAI_API_KEY is not configured with a real "
                "key and rules could not infer target_column."
            )
        return _fallback_plan(
            fallback,
            "Agent workflow is enabled, but OPENAI_API_KEY is not configured with a real key.",
        )

    agent_context = _agent_context(df, profile, user_goal, target_column)
    planner = Agent(
        name="DataPilot Workflow Planner",
        instructions=(
            "You plan controlled tabular data analysis workflows. "
            "Choose a target column only from the provided column list. "
            "Infer whether the task is classification, regression, or eda_only. "
            "Prefer the user's explicit target column when valid. "
            "Do not invent columns, metrics, or data properties. "
            "Return concise structured output."
        ),
        model=settings.agent_model,
        output_type=AgentWorkflowPlan,
    )

    provider = OpenAIProvider(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        use_responses=False,
    )
    run_config = RunConfig(
        model_provider=provider,
        workflow_name="DataPilot task planning",
        tracing_disabled=True,
    )

    try:
        result = Runner.run_sync(planner, agent_context, run_config=run_config, max_turns=3)
        raw_plan = result.final_output
        if not isinstance(raw_plan, AgentWorkflowPlan):
            raise ValueError("Agent did not return the expected structured plan.")
        plan = raw_plan.model_dump()
        validated = _validate_agent_plan(df, plan, fallback)
        return {**validated, "source": "agent", "enabled": True}
    except Exception as exc:
        if fallback is None:
            raise ValueError(
                f"Agent planning failed and no rule fallback is available: {exc}"
            ) from exc
        return _fallback_plan(fallback, f"Agent planning failed, using rule fallback: {exc}")


def _try_rule_fallback(
    df: pd.DataFrame,
    user_goal: str,
    target_column: str | None,
) -> dict | None:
    try:
        return infer_task(df, user_goal=user_goal, target_column=target_column)
    except ValueError:
        return None


def _has_real_api_key() -> bool:
    return bool(
        settings.openai_api_key
        and settings.openai_api_key.strip()
        and settings.openai_api_key != "your_api_key_here"
    )


def _fallback_plan(fallback: dict, reason: str) -> dict:
    return {
        **fallback,
        "analysis_focus": ["data profiling", "eda", "model training", "evaluation"],
        "workflow_steps": [
            "profile_dataset",
            "infer_task",
            "run_eda",
            "train_models",
            "evaluate_model",
            "write_report",
        ],
        "confidence": 0.45,
        "reason": reason,
        "source": "rules",
        "enabled": settings.enable_agent_workflow,
    }


def _agent_context(
    df: pd.DataFrame,
    profile: dict,
    user_goal: str,
    target_column: str | None,
) -> str:
    sample_rows = df.head(5).where(pd.notna(df.head(5)), None).to_dict(orient="records")
    column_stats = []
    for col in df.columns:
        series = df[col]
        examples = [str(value) for value in series.dropna().head(3).tolist()]
        column_stats.append(
            {
                "name": col,
                "type": profile["column_types"].get(col),
                "missing_rate": profile["missing_rates"].get(col),
                "unique_values": int(series.nunique(dropna=True)),
                "examples": examples,
            }
        )

    payload = {
        "user_goal": user_goal,
        "user_selected_target_column": target_column,
        "dataset_shape": {"rows": profile["rows"], "columns": profile["columns"]},
        "columns": column_stats,
        "target_candidates_from_rules": profile["target_candidates"],
        "sample_rows": sample_rows,
        "allowed_task_types": ["classification", "regression", "eda_only"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _validate_agent_plan(df: pd.DataFrame, plan: dict, fallback: dict | None) -> dict:
    target_column = plan.get("target_column")
    if not target_column or target_column not in df.columns:
        if fallback is None:
            raise ValueError("Agent did not choose a valid target column.")
        target_column = fallback["target_column"]

    checked = infer_task(df, user_goal="", target_column=target_column)
    task_type = plan.get("task_type")
    if task_type not in {"classification", "regression", "eda_only"}:
        task_type = checked["task_type"]
    if task_type == "eda_only":
        task_type = checked["task_type"]

    metric = plan.get("metric") or checked["metric"]
    if task_type != checked["task_type"]:
        task_type = checked["task_type"]
        metric = checked["metric"]

    return {
        **plan,
        "target_column": target_column,
        "task_type": task_type,
        "metric": metric,
        "validated_by_rules": True,
    }
