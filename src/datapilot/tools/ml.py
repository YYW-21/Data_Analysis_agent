from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_and_evaluate(
    df: pd.DataFrame,
    task_type: str,
    target_column: str,
    model_dir: Path,
    exclude_columns: list[str] | None = None,
) -> dict:
    model_dir.mkdir(parents=True, exist_ok=True)
    cleaned = df.drop_duplicates().copy()
    cleaned = cleaned.dropna(subset=[target_column])

    y = cleaned[target_column]
    x = cleaned.drop(columns=[target_column])
    excluded = [col for col in exclude_columns or [] if col in x.columns]
    if excluded:
        x = x.drop(columns=excluded)
    x = x.drop(columns=[col for col in x.columns if x[col].nunique(dropna=True) <= 1])

    datetime_cols = [
        col for col in x.columns if pd.api.types.is_datetime64_any_dtype(x[col])
    ]
    for col in x.columns:
        if x[col].dtype == "object":
            parsed = pd.to_datetime(x[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                x[col] = parsed
                datetime_cols.append(col)

    for col in datetime_cols:
        x[f"{col}_year"] = x[col].dt.year
        x[f"{col}_month"] = x[col].dt.month
        x[f"{col}_day"] = x[col].dt.day
        x = x.drop(columns=[col])

    numeric_features = x.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [
        col for col in x.columns if col not in numeric_features
    ]

    if not numeric_features and not categorical_features:
        raise ValueError("No usable feature columns after preprocessing.")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    stratify = y if task_type == "classification" and y.nunique() > 1 else None
    if stratify is not None and y.value_counts().min() < 2:
        stratify = None

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=stratify
    )

    candidates = _candidate_models(task_type)
    results = []
    for name, estimator in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = _metrics(task_type, y_test, predictions)
        results.append({"name": name, "pipeline": pipeline, "metrics": metrics})

    score_key = "f1_macro" if task_type == "classification" else "r2"
    best = max(results, key=lambda item: item["metrics"][score_key])
    model_path = model_dir / f"{best['name']}.joblib"
    joblib.dump(best["pipeline"], model_path)

    return {
        "best_model": best["name"],
        "model_path": str(model_path),
        "metrics": best["metrics"],
        "candidate_metrics": [
            {"model": item["name"], "metrics": item["metrics"]} for item in results
        ],
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "dropped": [target_column, *excluded],
        },
    }


def _candidate_models(task_type: str) -> dict:
    if task_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        }
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }


def _metrics(task_type: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    if task_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }
