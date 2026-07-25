# House Price Prediction

Complete Linear Regression project for predicting house prices from the provided housing dataset.

## Project Checklist

- Load the housing dataset from `data/house_prices.csv`.
- Clean the data, remove duplicates, normalize categories, validate numeric fields, and create `HouseAge`.
- Explore/save cleaned data and selected features.
- Split train/test data and train a Linear Regression model.
- Evaluate the model with RMSE, MAE, and R-squared.
- Interpret coefficients and save a coefficient report.
- Save the trained model and show example predictions.
- Open a Streamlit dashboard to view the dataset, metrics, plots, coefficients, and custom predictions.

## Project Structure

```text
.
|-- app.py
|-- data/
|   `-- house_prices.csv
|-- models/
|   `-- house_price_linear_regression.joblib
|-- reports/
|   |-- cleaned_house_prices.csv
|   |-- coefficients.csv
|   |-- example_predictions.csv
|   |-- metrics.json
|   |-- model_summary.md
|   `-- figures/
|-- src/
|   |-- data_processing.py
|   |-- predict.py
|   `-- train_model.py
|-- requirements.txt
`-- README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train the Model

```powershell
python src\train_model.py
```

The training script saves:

- `models/house_price_linear_regression.joblib`
- `reports/metrics.json`
- `reports/coefficients.csv`
- `reports/model_summary.md`
- `reports/example_predictions.csv`
- `reports/figures/actual_vs_predicted.png`
- `reports/figures/residuals.png`

## Open the Dashboard

```powershell
streamlit run app.py
```

Streamlit opens a local browser page, usually at:

```text
http://localhost:8501
```

## Make Example Predictions From Terminal

Run the built-in examples:

```powershell
python src\predict.py
```

Predict from your own CSV:

```powershell
python src\predict.py --input data\new_houses.csv --output reports\new_predictions.csv
```

Your prediction CSV should include these columns:

```text
Area,Bedrooms,Bathrooms,Floors,YearBuilt,Location,Condition,Garage
```

## Model Notes

The model uses these selected features:

```text
Area, Bedrooms, Bathrooms, Floors, HouseAge, Location, Condition, Garage
```

`YearBuilt` is converted into `HouseAge` using reference year `2026`. Categorical features are one-hot encoded with the first category dropped, so categorical coefficients compare each shown category with its omitted baseline.
