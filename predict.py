"""First program, used to predict the price of a car
    for a given mileage. This one use the model trained
    by the second program."""


import sys
import json
from train import load_csv
from pathlib import Path
import matplotlib.pyplot as plt


def get_mileage() -> float:
    """get the user input mileage, and return it."""
    while True:
        try:
            mileage = input("Enter a mileage: ").strip()

            if not mileage:
                raise AssertionError("input can not be empty.")
            n = float(mileage)
            if n != n or n == float("inf") or n == float("-inf"):
                raise ValueError("invalid input.")
            if n < 0:
                raise ValueError("mileage must be a positiv number.")
            if n > 2_000_000:
                raise ValueError("mileage is too high (max 2 000 000 km).")

            return n

        except (AssertionError, ValueError) as error:
            print(f"{type(error).__name__}: {error}")
            continue

        except (EOFError, KeyboardInterrupt):
            print("input canceled.")
            sys.exit(0)


def get_Price(df: dict, mileage: float):
    """get the estimate price for the mileage inputed by user."""
    mileageNorm = (mileage - df["min_km"]) / (df["max_km"] - df["min_km"])
    estimatedPrice = df["theta0"] + (df["theta1"] * mileageNorm)
    print(estimatedPrice)


def load_parameters(filepath: str) -> dict:
    """Load theta0, theta1, min_km, and max_km from a JSON file.
    If the file doesn't exist, return default values (theta0=0, theta1=0)."""
    path = Path(filepath)
    if not path.exists():
        print(
            "Notice: 'save_data.json' not found."
            "Using default parameters (0, 0)."
            )
        return {"theta0": 0.0, "theta1": 0.0, "min_km": 0.0, "max_km": 1.0}

    try:
        with open(path, "r") as f:
            data = json.load(f)
            required_keys = {"theta0", "theta1", "min_km", "max_km"}
            path.unlink(missing_ok=True)
            if not required_keys.issubset(data.keys()):
                raise ValueError("JSON file is missing required parameters.")
            return data

    except json.JSONDecodeError:
        print(
            "Error: 'save_data.json' is corrupted or malformed."
            "Using default parameters."
            )
        path.unlink(missing_ok=True)
        return {"theta0": 0.0, "theta1": 0.0, "min_km": 0.0, "max_km": 1.0}
    except (PermissionError, Exception) as error:
        print(f"Error loading parameters: {error}")
        path.unlink(missing_ok=True)
        sys.exit(1)


def display_graph(dataPath: str, data: dict,
                  mileage: float, estimated_price: float):
    """Display the data into a graph and plotting the resulting from the
    linear regression into the same graph."""
    df = load_csv(dataPath)
    x = df["km"]
    y = df["price"]

    line_x = [data["min_km"], data["max_km"]]
    line_y = [
        data["theta0"] + data["theta1"] *
        ((value - data["min_km"]) / (data["max_km"] - data["min_km"]))
        for value in line_x
    ]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="blue", alpha=0.6, label="Data from CSV")
    plt.plot(
        line_x, line_y, color="red", linewidth=2, label="Regression line"
    )
    plt.scatter(
        [mileage], [estimated_price], color="green", s=100, zorder=5,
        label=f"Your prediction ({estimated_price:,.0f} €)"
    )
    plt.title("Linear regression - Price of cars in fonction of mileage")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Graph display canceled.")
        sys.exit(0)


def main():
    """Main function that run the prediction script."""

    mileage = get_mileage()
    data = load_parameters("save_data.json")

    if data["max_km"] == data["min_km"]:
        km_norm = 0.0
    else:
        km_norm = (mileage - data["min_km"]) / (data["max_km"] -
                                                data["min_km"])

    estimated_price = data["theta0"] + (data["theta1"] * km_norm)

    if estimated_price < 0:
        estimated_price = 0.0

    print(
        f"\nEstimated Price for {mileage:,.0f}"
        f"km : {estimated_price:,.2f} €"
        )
    display_graph("data.csv", data, mileage, estimated_price)


if __name__ == "__main__":
    main()
