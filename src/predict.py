from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from data_processing import FEATURE_COLUMNS, clean_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "house_price_linear_regression.joblib"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "new_predictions.csv"


def default_examples() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Area": 1850,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "Floors": 2,
                "YearBuilt": 2010,
                "Location": "Suburban",
                "Condition": "Good",
                "Garage": "Yes",
            },
            {
                "Area": 3100,
                "Bedrooms": 4,
                "Bathrooms": 3,
                "Floors": 2,
                "YearBuilt": 1998,
                "Location": "Downtown",
                "Condition": "Excellent",
                "Garage": "No",
            },
            {
                "Area": 1250,
                "Bedrooms": 2,
                "Bathrooms": 1,
                "Floors": 1,
                "YearBuilt": 1975,
                "Location": "Rural",
                "Condition": "Fair",
                "Garage": "Yes",
            },
        ]
    )


def load_model(model_path: Path):
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"]
    return artifact


def predict(input_path: Path | None, model_path: Path, output_path: Path) -> pd.DataFrame:
    raw_data = pd.read_csv(input_path) if input_path else default_examples()
    cleaned_data = clean_data(raw_data, require_target=False)

    pipeline = load_model(model_path)
    predictions = pipeline.predict(cleaned_data[FEATURE_COLUMNS])

    results = raw_data.copy()
    results["PredictedPrice"] = predictions.round(2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict house prices with the saved model.")
    parser.add_argument("--input", type=Path, default=None, help="Optional CSV with new houses.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Saved model path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Prediction CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = predict(args.input, args.model, args.output)
    print(results.to_string(index=False))
    print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
