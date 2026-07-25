from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data_processing import FEATURE_COLUMNS, clean_data, load_dataset  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "house_prices.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_linear_regression.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
COEFFICIENTS_PATH = REPORTS_DIR / "coefficients.csv"
EXAMPLE_PREDICTIONS_PATH = REPORTS_DIR / "example_predictions.csv"


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="H",
    layout="wide",
)


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_data
def get_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"]
    return artifact


st.title("House Price Prediction")
st.caption("Linear Regression model trained on area, room counts, house age, location, condition, and garage availability.")

if not DATA_PATH.exists():
    st.error("Dataset not found. Put the CSV at data/house_prices.csv.")
    st.stop()

data = get_dataset()
metrics = get_metrics()
model = get_model()

metric_cols = st.columns(4)
metric_cols[0].metric("Rows", f"{len(data):,}")
metric_cols[1].metric("Average Price", f"{data['Price'].mean():,.0f}")
metric_cols[2].metric("RMSE", f"{metrics.get('rmse', 0):,.2f}" if metrics else "Train first")
metric_cols[3].metric("R-squared", f"{metrics.get('r2', 0):.4f}" if metrics else "Train first")

tab_overview, tab_model, tab_predict = st.tabs(["Dataset", "Model Output", "Predict Price"])

with tab_overview:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(data.head(25), use_container_width=True)
    with right:
        st.subheader("Feature Summary")
        st.write(data[["Area", "Bedrooms", "Bathrooms", "Floors", "HouseAge", "Price"]].describe())

    chart_data = data[["Area", "Price"]].rename(columns={"Area": "area", "Price": "price"})
    st.subheader("Area vs Price")
    st.scatter_chart(chart_data, x="area", y="price", use_container_width=True)

with tab_model:
    if not metrics:
        st.info("Run `python src\\train_model.py` first to generate metrics, plots, coefficients, and example predictions.")
    else:
        st.subheader("Evaluation Metrics")
        st.json(metrics)

    coefficients = read_csv_if_exists(COEFFICIENTS_PATH)
    if not coefficients.empty:
        st.subheader("Largest Coefficients")
        top_coefficients = coefficients.head(12).set_index("feature")[["coefficient"]]
        st.bar_chart(top_coefficients, use_container_width=True)
        st.dataframe(coefficients, use_container_width=True)

    example_predictions = read_csv_if_exists(EXAMPLE_PREDICTIONS_PATH)
    if not example_predictions.empty:
        st.subheader("Example Predictions")
        st.dataframe(example_predictions, use_container_width=True)

    actual_vs_predicted = REPORTS_DIR / "figures" / "actual_vs_predicted.png"
    residuals = REPORTS_DIR / "figures" / "residuals.png"
    if actual_vs_predicted.exists() or residuals.exists():
        image_cols = st.columns(2)
        if actual_vs_predicted.exists():
            image_cols[0].image(str(actual_vs_predicted), caption="Actual vs Predicted")
        if residuals.exists():
            image_cols[1].image(str(residuals), caption="Residuals")

with tab_predict:
    if model is None:
        st.warning("Train the model before making dashboard predictions.")
    else:
        st.subheader("Enter House Details")
        form_cols = st.columns(4)
        area = form_cols[0].number_input("Area", min_value=100, max_value=10000, value=1850, step=50)
        bedrooms = form_cols[1].number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
        bathrooms = form_cols[2].number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        floors = form_cols[3].number_input("Floors", min_value=1, max_value=5, value=2, step=1)

        form_cols = st.columns(4)
        year_built = form_cols[0].number_input("Year Built", min_value=1800, max_value=2026, value=2010, step=1)
        location = form_cols[1].selectbox("Location", sorted(data["Location"].dropna().unique()))
        condition = form_cols[2].selectbox("Condition", sorted(data["Condition"].dropna().unique()))
        garage = form_cols[3].selectbox("Garage", sorted(data["Garage"].dropna().unique()))

        row = pd.DataFrame(
            [
                {
                    "Area": area,
                    "Bedrooms": bedrooms,
                    "Bathrooms": bathrooms,
                    "Floors": floors,
                    "YearBuilt": year_built,
                    "Location": location,
                    "Condition": condition,
                    "Garage": garage,
                }
            ]
        )

        cleaned_row = clean_data(row, require_target=False)
        prediction = model.predict(cleaned_row[FEATURE_COLUMNS])[0]
        st.metric("Predicted House Price", f"{prediction:,.2f}")
