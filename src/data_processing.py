from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_COLUMN = "Price"
REFERENCE_YEAR = 2026

NUMERIC_FEATURES = ["Area", "Bedrooms", "Bathrooms", "Floors", "HouseAge"]
CATEGORICAL_FEATURES = ["Location", "Condition", "Garage"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RAW_NUMERIC_COLUMNS = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "YearBuilt",
    TARGET_COLUMN,
]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and apply project cleaning rules."""
    return clean_data(pd.read_csv(csv_path))


def clean_data(data: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Clean raw house records and create model-ready feature columns."""
    df = data.copy()
    df.columns = [column.strip() for column in df.columns]
    df = df.drop_duplicates()
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    for column in RAW_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip().str.title()

    if "Garage" in df.columns:
        df["Garage"] = df["Garage"].replace(
            {
                "Y": "Yes",
                "N": "No",
                "True": "Yes",
                "False": "No",
                "1": "Yes",
                "0": "No",
            }
        )

    if "YearBuilt" not in df.columns:
        raise ValueError("The dataset must include a YearBuilt column.")

    df["HouseAge"] = REFERENCE_YEAR - df["YearBuilt"]

    if require_target:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"The dataset must include a {TARGET_COLUMN} column.")
        df = df[df[TARGET_COLUMN].notna()]
        df = df[df[TARGET_COLUMN] > 0]

    if "Area" in df.columns:
        df = df[df["Area"].isna() | (df["Area"] > 0)]

    if "YearBuilt" in df.columns:
        valid_year = df["YearBuilt"].between(1800, REFERENCE_YEAR)
        df = df[df["YearBuilt"].isna() | valid_year]

    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {', '.join(missing_columns)}")

    return df.reset_index(drop=True)


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return selected model features and target values."""
    return data[FEATURE_COLUMNS], data[TARGET_COLUMN]
