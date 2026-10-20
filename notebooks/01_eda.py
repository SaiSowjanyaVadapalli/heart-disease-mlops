"""
Exploratory Data Analysis for the Heart Disease UCI dataset.

Run after scripts/download_data.py. Produces professional EDA visuals under
reports/figures/ and a short written summary at reports/EDA_SUMMARY.md.

    python notebooks/01_eda.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: safe for CI and non-GUI environments
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "heart_disease.csv"
FIG_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "EDA_SUMMARY.md"

sns.set_theme(style="whitegrid", palette="viridis")

FEATURE_LABELS = {
    "age": "Age (years)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "thalach": "Max Heart Rate Achieved",
    "oldpeak": "ST Depression (exercise vs rest)",
}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_PATH)

    lines = ["# Heart Disease Dataset -- EDA Summary", ""]
    lines.append(f"- Rows: **{len(df)}**, Columns: **{df.shape[1]}**")
    lines.append(f"- Missing values:\n\n```\n{df.isna().sum().to_string()}\n```")
    lines.append(
        f"\n- Descriptive statistics:\n\n```\n{df.describe().round(2).to_string()}\n```"
    )

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["target"].value_counts().sort_index()
    labels = ["No Disease (0)", "Disease (1)"]
    sns.barplot(x=labels, y=counts.values, ax=ax, hue=labels, legend=False)
    ax.set_title("Target Class Balance")
    ax.set_ylabel("Patient Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_balance.png", dpi=150)
    plt.close(fig)

    # 2. Histograms of key continuous features
    numeric_cols = list(FEATURE_LABELS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#2a6f97")
        ax.set_title(FEATURE_LABELS[col])
        ax.set_xlabel("")
    for ax in axes.flat[len(numeric_cols) :]:
        ax.axis("off")
    fig.suptitle("Distributions of Continuous Clinical Features", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_histograms.png", dpi=150)
    plt.close(fig)

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True
    )
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 4. Target vs key features (boxplots)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ["age", "thalach", "oldpeak"]):
        sns.boxplot(x="target", y=col, data=df, ax=ax, hue="target", legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Disease", "Disease"])
        ax.set_title(f"{FEATURE_LABELS[col]} by Diagnosis")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "target_vs_features.png", dpi=150)
    plt.close(fig)

    # 5. Categorical feature breakdown (chest pain type vs disease)
    fig, ax = plt.subplots(figsize=(7, 5))
    cp_target = pd.crosstab(df["cp"], df["target"])
    cp_target.plot(kind="bar", stacked=True, ax=ax, color=["#8ecae6", "#e63946"])
    ax.set_title("Chest Pain Type vs Disease Presence")
    ax.set_xlabel("Chest Pain Type (cp code)")
    ax.set_ylabel("Patient Count")
    ax.legend(["No Disease", "Disease"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chest_pain_vs_target.png", dpi=150)
    plt.close(fig)

    lines.append("\n## Figures generated")
    lines.append("- reports/figures/class_balance.png")
    lines.append("- reports/figures/feature_histograms.png")
    lines.append("- reports/figures/correlation_heatmap.png")
    lines.append("- reports/figures/target_vs_features.png")
    lines.append("- reports/figures/chest_pain_vs_target.png")

    top_corr = corr["target"].drop("target").abs().sort_values(ascending=False).head(5)
    lines.append("\n## Features most correlated with target (|corr|, top 5)")
    lines.append(f"```\n{top_corr.round(3).to_string()}\n```")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"EDA complete. Figures -> {FIG_DIR}")
    print(f"Summary -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
