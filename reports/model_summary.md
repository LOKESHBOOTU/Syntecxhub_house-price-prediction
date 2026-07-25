# House Price Linear Regression Summary

## Evaluation

- RMSE: 279,859.73
- MAE: 243,241.98
- R-squared: -0.0067

RMSE gives the typical size of prediction error in the same units as house price.
R-squared estimates how much variation in price is explained by the selected features.

## Coefficient Interpretation

Numeric coefficients estimate the price change for a one-unit increase in that feature, holding the other selected features constant.
Categorical coefficients compare a category with the omitted baseline category from one-hot encoding.

### Strongest Positive Coefficients

- Condition_Fair: 24,083.31
- Floors: 23,727.98
- Location_Suburban: 11,511.99
- Condition_Poor: 4,073.27
- Garage_Yes: 2,373.53

### Strongest Negative Coefficients

- Condition_Good: -12,941.04
- Location_Urban: -12,718.92
- Bathrooms: -9,662.25
- HouseAge: -117.61
- Area: -0.58
