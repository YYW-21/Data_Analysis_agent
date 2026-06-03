import warnings

import pandas as pd


def infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series.dropna().head(50), errors="coerce")
    if len(parsed) > 0 and parsed.notna().mean() > 0.8:
        return "datetime"
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    if unique_ratio < 0.2 or series.nunique(dropna=True) <= 30:
        return "categorical"
    return "text"


def profile_dataframe(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    column_types = {col: infer_column_type(df[col]) for col in df.columns}
    target_candidates = _target_candidates(df, column_types)
    warnings = []
    if len(df) < 50:
        warnings.append("Dataset has fewer than 50 rows; model evaluation may be unstable.")
    high_missing = [col for col, count in missing.items() if count / max(len(df), 1) > 0.4]
    if high_missing:
        warnings.append(f"High-missing columns detected: {', '.join(high_missing)}")

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_types": column_types,
        "missing_values": {col: int(count) for col, count in missing.items()},
        "missing_rates": {col: float(count / max(len(df), 1)) for col, count in missing.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "target_candidates": target_candidates,
        "warnings": warnings,
    }


def _target_candidates(df: pd.DataFrame, column_types: dict[str, str]) -> list[str]:
    strong_names = {
        "target",
        "label",
        "class",
        "y",
        "survived",
        "churn",
        "price",
        "sales",
        "revenue",
        "score",
        "rating",
        "outcome",
        "result",
    }
    soft_keywords = [
        "target",
        "label",
        "class",
        "churn",
        "price",
        "score",
        "rating",
        "outcome",
        "result",
        "survive",
        "是否",
        "标签",
        "结果",
        "目标",
        "价格",
        "评分",
        "流失",
    ]

    candidates: list[str] = []
    for col in df.columns:
        normalized = col.lower()
        if normalized in strong_names or any(keyword in normalized for keyword in soft_keywords):
            candidates.append(col)

    for col in df.columns:
        if col in candidates:
            continue
        unique_values = df[col].nunique(dropna=True)
        missing_rate = df[col].isna().mean()
        if missing_rate > 0.5 or unique_values < 2:
            continue
        if column_types[col] in {"boolean", "categorical"} and unique_values <= 20:
            candidates.append(col)
        elif column_types[col] == "numeric" and 2 <= unique_values <= min(50, max(len(df) // 2, 2)):
            candidates.append(col)

    return candidates
