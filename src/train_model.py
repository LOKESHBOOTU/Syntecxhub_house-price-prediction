from __future__ import annotations

import argparse
import json
from datetime import datetime
from math import sqrt
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_processing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    clean_data,
    load_dataset,
    split_features_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "house_prices.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "house_price_linear_regression.joblib"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_FIGURES_DIR = DEFAULT_REPORTS_DIR / "figures"
RANDOM_STATE = 42


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def calculate_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "rmse": calculate_rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    readable_names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in feature_names
    ]

    coefficients = pd.DataFrame(
        {
            "feature": readable_names,
            "coefficient": model.coef_,
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    return coefficients.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def save_figures(
    y_true: pd.Series,
    y_pred: pd.Series,
    figures_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    residuals = y_true - y_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.65)
    min_price = min(y_true.min(), y_pred.min())
    max_price = max(y_true.max(), y_pred.max())
    plt.plot([min_price, max_price], [min_price, max_price], color="red", linewidth=2)
    plt.title("Actual vs Predicted House Prices")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.tight_layout()
    plt.savefig(figures_dir / "actual_vs_predicted.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=25, edgecolor="black")
    plt.title("Prediction Residuals")
    plt.xlabel("Actual price - predicted price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "residuals.png", dpi=160)
    plt.close()


def write_model_summary(
    metrics: dict[str, float],
    coefficients: pd.DataFrame,
    reports_dir: Path,
) -> None:
    top_positive = coefficients.sort_values("coefficient", ascending=False).head(5)
    top_negative = coefficients.sort_values("coefficient", ascending=True).head(5)

    lines = [
        "# House Price Linear Regression Summary",
        "",
        "## Evaluation",
        "",
        f"- RMSE: {metrics['rmse']:,.2f}",
        f"- MAE: {metrics['mae']:,.2f}",
        f"- R-squared: {metrics['r2']:.4f}",
        "",
        "RMSE gives the typical size of prediction error in the same units as house price.",
        "R-squared estimates how much variation in price is explained by the selected features.",
        "",
        "## Coefficient Interpretation",
        "",
        "Numeric coefficients estimate the price change for a one-unit increase in that feature, holding the other selected features constant.",
        "Categorical coefficients compare a category with the omitted baseline category from one-hot encoding.",
        "",
        "### Strongest Positive Coefficients",
        "",
    ]

    for _, row in top_positive.iterrows():
        lines.append(f"- {row['feature']}: {row['coefficient']:,.2f}")

    lines.extend(["", "### Strongest Negative Coefficients", ""])

    for _, row in top_negative.iterrows():
        lines.append(f"- {row['feature']}: {row['coefficient']:,.2f}")

    (reports_dir / "model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(data_path: Path, model_path: Path, reports_dir: Path) -> dict[str, float]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_data = load_dataset(data_path)
    cleaned_data.to_csv(reports_dir / "cleaned_house_prices.csv", index=False)

    x, y = split_features_target(cleaned_data)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = evaluate_model(y_test, predictions)

    coefficients = get_coefficients(pipeline)
    coefficients.to_csv(reports_dir / "coefficients.csv", index=False)

    examples = x_test.copy()
    examples["ActualPrice"] = y_test
    examples["PredictedPrice"] = predictions
    examples["PredictionError"] = examples["ActualPrice"] - examples["PredictedPrice"]
    examples.head(12).to_csv(reports_dir / "example_predictions.csv", index=False)

    metrics_payload = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "rows_after_cleaning": int(len(cleaned_data)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        **metrics,
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    write_model_summary(metrics, coefficients, reports_dir)
    save_figures(y_test, predictions, reports_dir / "figures")

    artifact = {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metrics": metrics_payload,
    }
    joblib.dump(artifact, model_path)

    return metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a house price Linear Regression model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to CSV data.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path for saved model.")
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS_DIR, help="Reports output folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(args.data, args.model, args.reports)
    print("Training complete.")
    print(f"Rows after cleaning: {metrics['rows_after_cleaning']}")
    print(f"RMSE: {metrics['rmse']:,.2f}")
    print(f"MAE: {metrics['mae']:,.2f}")
    print(f"R-squared: {metrics['r2']:.4f}")
    print(f"Model saved to: {args.model}")


if __name__ == "__main__":
    main()
