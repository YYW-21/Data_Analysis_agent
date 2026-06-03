import pandas as pd


def infer_task(df: pd.DataFrame, user_goal: str, target_column: str | None) -> dict:
    if not target_column:
        target_column = choose_target_column(df, user_goal)
    if not target_column:
        raise ValueError("Could not infer target column. Please provide target_column.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")

    target = df[target_column].dropna()
    if target.empty:
        raise ValueError("Target column has no non-missing values.")

    if pd.api.types.is_numeric_dtype(target) and target.nunique() > 10:
        return {"task_type": "regression", "target_column": target_column, "metric": "r2"}
    return {"task_type": "classification", "target_column": target_column, "metric": "f1_macro"}


def choose_target_column(df: pd.DataFrame, user_goal: str) -> str | None:
    goal = user_goal.lower()
    for col in df.columns:
        normalized = col.lower()
        if normalized in goal:
            return col
    for candidate in ["target", "label", "class", "y", "survived", "churn", "price"]:
        for col in df.columns:
            if col.lower() == candidate:
                return col
    return None

