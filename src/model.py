# src/model.py

import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate(X, y):
    """
    Split data chronologically, train a LinearRegression model,
    print MSE/R², and show a residuals plot.
    """
    # Chronological split: first 80% for train, last 20% for test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    coef = model.coef_[0]
    intercept = model.intercept_

    print(f"✅ Model trained on {len(X_train)} samples, tested on {len(X_test)} samples")
    print(f"MSE:        {mse:.4f}")
    print(f"R² Score:  {r2:.4f}")
    print(f"Coefficient: {coef:.4f}  (impact per % unemployment)")
    print(f"Intercept:   {intercept:.4f}")

    # Residual plot
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, (y_test - y_pred), alpha=0.7)
    plt.axhline(0, linestyle='--', linewidth=1, color='black')
    plt.xlabel("Predicted GDP Growth (%)")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted GDP Growth")
    plt.tight_layout()
    plt.show()
