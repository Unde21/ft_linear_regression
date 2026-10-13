"""Program to evaluate the performance and accuracy
    of the trained linear regression model."""


import json
import sys
from train import load_csv
from pathlib import Path


def main():
    """Main function to compute evaluation metrics (MSE, RMSE, R²)."""

    path = Path("save_data.json")
    if not path.exists():
        print(
            "Error: 'save_data.json' not found."
            "Please run 'train.py' first."
            )
        return

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, PermissionError, Exception) as error:
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)

    df = load_csv("data.csv")
    x = df["km"]
    y_real = df["price"]

    if data["max_km"] == data["min_km"]:
        x_norm = 0.0 * x
    else:
        x_norm = (x - data["min_km"]) / (data["max_km"] - data["min_km"])

    y_pred = data["theta0"] + (data["theta1"] * x_norm)

    mse = ((y_pred - y_real) ** 2).mean()
    rmse = mse ** 0.5

    ss_total = ((y_real - y_real.mean()) ** 2).sum()
    ss_residual = ((y_pred - y_real) ** 2).sum()
    r2_score = 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0

    print("=" * 45)
    print("      REGRESSION MODEL EVALUATION")
    print("=" * 45)
    print(f"-> Tested samples count    : {len(df)}")
    print(f"-> MSE (Mean Squared Error): {mse:,.2f}")
    print(f"-> RMSE (Average error)    : ± {rmse:,.2f} €")
    print(f"-> R² Score (Model quality): {r2_score * 100:.2f}%")
    print("=" * 45)


if __name__ == "__main__":
    main()
