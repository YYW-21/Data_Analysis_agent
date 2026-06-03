import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


def train_and_evaluate(
    df: pd.DataFrame,
    task_type: str,
    target_column: str,
    model_dir: Path,
    processed_dir: Path | None = None,
    artifact_dir: Path | None = None,
) -> dict:
    model_dir.mkdir(parents=True, exist_ok=True)
    if processed_dir:
        processed_dir.mkdir(parents=True, exist_ok=True)
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    original_rows = len(df)
    cleaned = df.drop_duplicates().copy()
    cleaned = cleaned.dropna(subset=[target_column])

    y = cleaned[target_column]
    x = cleaned.drop(columns=[target_column])
    dropped_constant_columns = [col for col in x.columns if x[col].nunique(dropna=True) <= 1]
    x = x.drop(columns=dropped_constant_columns)

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

    processed_paths = _save_processed_outputs(
        x=x,
        y=y,
        processed_dir=processed_dir,
        target_column=target_column,
        original_rows=original_rows,
        cleaned_rows=len(cleaned),
        duplicate_rows=int(df.duplicated().sum()),
        dropped_target_missing_rows=int(original_rows - len(df.dropna(subset=[target_column]))),
        dropped_constant_columns=dropped_constant_columns,
        datetime_columns=datetime_cols,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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

    candidates = _candidate_models(task_type, y_train)
    results = []
    failed_models = []
    for name, estimator in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        try:
            y_fit = y_train
            target_encoder = None
            if task_type == "classification" and name == "xgboost":
                target_encoder = LabelEncoder()
                y_fit = target_encoder.fit_transform(y_train)

            pipeline.fit(x_train, y_fit)
            predictions = pipeline.predict(x_test)
            if target_encoder is not None:
                predictions = target_encoder.inverse_transform(predictions.astype(int))

            metrics = _metrics(task_type, y_test, predictions)
            results.append(
                {
                    "name": name,
                    "pipeline": pipeline,
                    "target_encoder": target_encoder,
                    "metrics": metrics,
                }
            )
        except Exception as exc:
            failed_models.append({"model": name, "error": str(exc)})

    if not results:
        errors = "; ".join(f"{item['model']}: {item['error']}" for item in failed_models)
        raise ValueError(f"All candidate models failed. {errors}")

    score_key = "f1_macro" if task_type == "classification" else "r2"
    best = max(results, key=lambda item: item["metrics"][score_key])
    model_path = model_dir / f"{best['name']}.joblib"
    artifact = {"pipeline": best["pipeline"], "target_encoder": best["target_encoder"]}
    joblib.dump(artifact, model_path)
    evaluation_artifacts = _save_evaluation_artifacts(
        task_type=task_type,
        pipeline=best["pipeline"],
        target_encoder=best["target_encoder"],
        x_test=x_test,
        y_test=y_test,
        artifact_dir=artifact_dir,
    )
    feature_importance = _save_feature_importance(
        pipeline=best["pipeline"],
        artifact_dir=artifact_dir,
    )

    return {
        "best_model": best["name"],
        "model_path": str(model_path),
        "metrics": best["metrics"],
        "candidate_metrics": [
            {"model": item["name"], "metrics": item["metrics"]} for item in results
        ],
        "failed_models": failed_models,
        "processed": processed_paths,
        "evaluation_artifacts": evaluation_artifacts,
        "feature_importance": feature_importance,
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "dropped": [target_column, *dropped_constant_columns],
        },
    }


def predict_with_model(model_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    target_encoder = artifact.get("target_encoder")
    predictions = pipeline.predict(df)
    if target_encoder is not None:
        predictions = target_encoder.inverse_transform(predictions.astype(int))
    result = df.copy()
    result["prediction"] = predictions
    return result


def _save_processed_outputs(
    x: pd.DataFrame,
    y: pd.Series,
    processed_dir: Path | None,
    target_column: str,
    original_rows: int,
    cleaned_rows: int,
    duplicate_rows: int,
    dropped_target_missing_rows: int,
    dropped_constant_columns: list[str],
    datetime_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict:
    if processed_dir is None:
        return {}

    imputed_x, imputation_details = _impute_feature_frame(
        x=x,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    cleaned_preview = imputed_x.copy()
    cleaned_preview[target_column] = y
    cleaned_path = processed_dir / "cleaned.csv"
    feature_preview_path = processed_dir / "feature_preview.csv"
    summary_path = processed_dir / "preprocessing_summary.json"

    cleaned_preview.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    imputed_x.head(50).to_csv(feature_preview_path, index=False, encoding="utf-8-sig")

    summary = {
        "target_column": target_column,
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "duplicate_rows_removed": duplicate_rows,
        "target_missing_rows_removed": dropped_target_missing_rows,
        "dropped_constant_columns": dropped_constant_columns,
        "datetime_columns_expanded": datetime_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "imputation": {
            "numeric": "median",
            "categorical": "most_frequent",
            "fallback_numeric": 0,
            "fallback_categorical": "unknown",
            "details": imputation_details,
        },
        "encoding": {
            "categorical": "one_hot",
        },
        "scaling": {
            "numeric": "standard_scaler",
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "cleaned_csv": str(cleaned_path),
        "feature_preview_csv": str(feature_preview_path),
        "preprocessing_summary_json": str(summary_path),
    }


def _impute_feature_frame(
    x: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, dict]:
    imputed = x.copy()
    details = {
        "numeric": {},
        "categorical": {},
    }

    for col in numeric_features:
        missing_count = int(imputed[col].isna().sum())
        fill_value = imputed[col].median()
        if pd.isna(fill_value):
            fill_value = 0
        imputed[col] = imputed[col].fillna(fill_value)
        details["numeric"][col] = {
            "missing_count": missing_count,
            "fill_value": float(fill_value),
        }

    for col in categorical_features:
        missing_count = int(imputed[col].isna().sum())
        mode = imputed[col].mode(dropna=True)
        fill_value = str(mode.iloc[0]) if not mode.empty else "unknown"
        imputed[col] = imputed[col].fillna(fill_value)
        details["categorical"][col] = {
            "missing_count": missing_count,
            "fill_value": fill_value,
        }

    return imputed, details


def _save_evaluation_artifacts(
    task_type: str,
    pipeline: Pipeline,
    target_encoder: LabelEncoder | None,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    artifact_dir: Path | None,
) -> list[str]:
    if artifact_dir is None:
        return []

    predictions = pipeline.predict(x_test)
    if target_encoder is not None:
        predictions = target_encoder.inverse_transform(predictions.astype(int))

    if task_type == "classification":
        return _save_classification_artifacts(y_test, predictions, artifact_dir)
    return _save_regression_artifacts(y_test, predictions, artifact_dir)


def _save_classification_artifacts(
    y_test: pd.Series,
    predictions: np.ndarray,
    artifact_dir: Path,
) -> list[str]:
    artifacts = []
    path = artifact_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    artifacts.append(str(path))
    return artifacts


def _save_regression_artifacts(
    y_test: pd.Series,
    predictions: np.ndarray,
    artifact_dir: Path,
) -> list[str]:
    artifacts = []

    predicted_path = artifact_dir / "predicted_vs_actual.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, predictions, alpha=0.8)
    min_value = min(float(np.min(y_test)), float(np.min(predictions)))
    max_value = max(float(np.max(y_test)), float(np.max(predictions)))
    ax.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")
    ax.set_title("Predicted vs Actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(predicted_path)
    plt.close(fig)
    artifacts.append(str(predicted_path))

    residual_path = artifact_dir / "residual_plot.png"
    residuals = np.asarray(y_test) - predictions
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(predictions, residuals, alpha=0.8)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Residual Plot")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(residual_path)
    plt.close(fig)
    artifacts.append(str(residual_path))

    return artifacts


def _save_feature_importance(pipeline: Pipeline, artifact_dir: Path | None) -> dict:
    if artifact_dir is None:
        return {}

    model = pipeline.named_steps["model"]
    importance = _extract_model_importance(model)
    if importance is None:
        return {}

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    size = min(len(feature_names), len(importance))
    importance_df = pd.DataFrame(
        {
            "feature": feature_names[:size],
            "importance": np.asarray(importance)[:size],
        }
    )
    importance_df["importance"] = importance_df["importance"].abs()
    importance_df = importance_df.sort_values("importance", ascending=False).head(20)

    csv_path = artifact_dir / "feature_importance.csv"
    png_path = artifact_dir / "feature_importance.png"
    importance_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

    return {
        "csv": str(csv_path),
        "plot": str(png_path),
        "top_features": importance_df.head(10).to_dict(orient="records"),
    }


def _extract_model_importance(model) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "coef_"):
        coefficient = np.asarray(model.coef_)
        if coefficient.ndim == 2:
            return np.mean(np.abs(coefficient), axis=0)
        return np.abs(coefficient)
    return None


def _candidate_models(task_type: str, y_train: pd.Series) -> dict:
    if task_type == "classification":
        xgb_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 1,
            "eval_metric": "logloss" if y_train.nunique() <= 2 else "mlogloss",
        }
        if y_train.nunique() > 2:
            xgb_params["objective"] = "multi:softprob"
            xgb_params["num_class"] = int(y_train.nunique())
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "extra_trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
            "xgboost": XGBClassifier(**xgb_params),
        }
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
        "xgboost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ),
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
