"""Exploratory Data Analysis (EDA) for the Smart Expense Assistant dataset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "expenses.csv"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"


def run_eda(
    data_path: Path = DATA_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, object]:
    """Run dataset overview, quality checks, summary statistics, and save plots."""
    df = pd.read_csv(data_path)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df["description_length_chars"] = df["description"].astype(str).str.len()
    df["description_length_words"] = df["description"].astype(str).str.split().str.len()

    summary = {
        "total_rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_descriptions": int(df["description"].str.lower().str.strip().duplicated().sum()),
        "category_distribution": {k: int(v) for k, v in df["category"].value_counts().to_dict().items()},
        "amount_summary": {
            "min": round(float(df["amount"].min()), 2),
            "median": round(float(df["amount"].median()), 2),
            "mean": round(float(df["amount"].mean()), 2),
            "max": round(float(df["amount"].max()), 2),
            "std": round(float(df["amount"].std()), 2),
        },
        "amount_by_category": df.groupby("category")["amount"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
        .to_dict(orient="index"),
        "description_length_chars": {
            "min": int(df["description_length_chars"].min()),
            "mean": round(float(df["description_length_chars"].mean()), 2),
            "max": int(df["description_length_chars"].max()),
        },
        "description_length_words": {
            "min": int(df["description_length_words"].min()),
            "mean": round(float(df["description_length_words"].mean()), 2),
            "max": int(df["description_length_words"].max()),
        },
    }

    sns.set_theme(style="whitegrid")

    # 1. Category distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["Food", "Travel", "Shopping", "Utilities"]
    sns.countplot(data=df, x="category", order=order, hue="category", palette="viridis", legend=False, ax=ax)
    ax.set_title("Expense Category Distribution")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Records")
    for container in ax.containers:
        ax.bar_label(container)
    fig.tight_layout()
    fig.savefig(figures_dir / "category_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Amount distribution by category
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x="category", y="amount", order=order, hue="category", palette="Set2", legend=False, ax=ax)
    ax.set_title("Transaction Amount Distribution by Category (INR)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Amount (INR)")
    fig.tight_layout()
    fig.savefig(figures_dir / "amount_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Description word count distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(data=df, x="description_length_words", bins=12, kde=True, color="#2b6cb0", ax=ax)
    ax.set_title("Expense Description Length Distribution (Words)")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(figures_dir / "description_length.png", dpi=150)
    plt.close(fig)

    return summary


if __name__ == "__main__":
    stats = run_eda()
    print(json.dumps(stats, indent=2))
