from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda(df: pd.DataFrame, target_column: str, artifact_dir: Path) -> list[str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []

    missing_path = artifact_dir / "missing_values.png"
    missing = df.isna().mean().sort_values(ascending=False).head(30)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing.values, y=missing.index)
    plt.title("Missing Value Rate")
    plt.xlabel("Rate")
    plt.tight_layout()
    plt.savefig(missing_path)
    plt.close()
    artifacts.append(str(missing_path))

    target_path = artifact_dir / "target_distribution.png"
    plt.figure(figsize=(8, 5))
    if pd.api.types.is_numeric_dtype(df[target_column]) and df[target_column].nunique() > 20:
        sns.histplot(df[target_column].dropna(), kde=True)
    else:
        df[target_column].astype(str).value_counts().head(20).plot(kind="bar")
    plt.title(f"Target Distribution: {target_column}")
    plt.tight_layout()
    plt.savefig(target_path)
    plt.close()
    artifacts.append(str(target_path))

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        corr_path = artifact_dir / "correlation_heatmap.png"
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(numeric_only=True), cmap="coolwarm", center=0)
        plt.title("Numeric Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(corr_path)
        plt.close()
        artifacts.append(str(corr_path))

    return artifacts

