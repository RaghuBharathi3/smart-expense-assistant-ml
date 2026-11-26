"""Data cleaning, text normalization, and stratified splitting for expense data."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2


def clean_text(text: str) -> str:
    """Normalize an expense description for TF-IDF feature extraction.

    Converts to lowercase, strips leading/trailing whitespace, removes
    non-essential punctuation while preserving alphanumeric characters and
    hyphens, and collapses repeated spaces.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str for text normalization, got {type(text).__name__}")

    normalized = text.lower().strip()
    normalized = re.sub(r"[^a-z0-9\s\-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def load_and_clean_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load CSV data, handle missing values, remove duplicates, and normalize text."""
    df = pd.read_csv(csv_path)

    required_cols = {"description", "category"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_cols)}")

    # Drop rows with missing description or category
    df = df.dropna(subset=["description", "category"]).copy()
    df["description"] = df["description"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    # Remove empty descriptions
    df = df[df["description"] != ""].copy()

    # Normalize text and remove case-insensitive duplicates
    df["clean_description"] = df["description"].apply(clean_text)
    df = df[df["clean_description"] != ""].copy()
    df = df.drop_duplicates(subset=["clean_description"]).reset_index(drop=True)

    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Perform a stratified 80:20 train-test split to prevent data leakage."""
    X = df["clean_description"]
    y = df["category"]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
