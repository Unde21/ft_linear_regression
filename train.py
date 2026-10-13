"""Second program, used to train the model. It will read a dataset file,
    and perform a linear regression on the data."""


import pandas as pd
import sys
import json


def load_csv(path: str) -> pd.DataFrame:
    """Function that receive Dataset in csv format,
    and return it as Pandas DataFrame if none error occured."""

    try:
        if not isinstance(path, str):
            raise TypeError("path must be a string.")
        if not path.lower().endswith((".csv")):
            raise ValueError("file extension must be .csv.")

        df = pd.read_csv(path)

        if "km" not in df.columns or "price" not in df.columns:
            raise KeyError("csv file must contain 'km' and 'price' colums.")
        if df.empty:
            raise ValueError("csv file is empty.")

        return df

    except (TypeError, ValueError, KeyError,
            FileNotFoundError, PermissionError,
            IsADirectoryError) as error:
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)

    except Exception as error:
        print(f"Unexpected error loading file: {error}")
        sys.exit(1)


def saveData(theta0: float, theta1: float, max_km: int, min_km: int) -> None:
    """Save theta0, theta1, and border km to file .json"""
    data = {
        "theta0": float(theta0),
        "theta1": float(theta1),
        "max_km": int(max_km),
        "min_km": int(min_km),
    }

    try:
        with open("save_data.json", "w") as f:
            json.dump(data, f, indent=4)

    except (PermissionError, OSError, IsADirectoryError) as error:
        print(f"{type(error).__name__}: Failed to save data: {error}")
        sys.exit(1)


def norm_value(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize km value to value between 0, 1."""
    df["km_norm"] = (df["km"] - df["km"].min()) / (df["km"].max()
                                                   - df["km"].min())
    return df


def learning(df: pd.DataFrame):
    """Train the model and perform a linear regression on the DataFrame."""
    theta0, theta1, tmp_theta0, tmp_theta1 = 0.0, 0.0, 0.0, 0.0
    learningRate = 0.5
    m = len(df)
    precision = 1e-6
    for x in range(2000):
        estimatePrice = theta0 + (theta1 * df["km_norm"])
        errors = estimatePrice - df["price"]
        tmp_theta0 = theta0 - (learningRate * (1 / m) * (errors.sum()))
        tmp_theta1 = theta1 - (learningRate * (1 / m) *
                               (errors * df["km_norm"]).sum())

        delta_theta0 = abs(theta0 - tmp_theta0)
        delta_theta1 = abs(theta1 - tmp_theta1)

        theta0, theta1 = tmp_theta0, tmp_theta1
        if delta_theta0 < precision and delta_theta1 < precision:
            print(f"convergence attempt at iteration {x}")
            break
    print(
        f"tmp_theta0: {tmp_theta0} tmp_theta1: {tmp_theta1},"
        f"{tmp_theta0 + tmp_theta1}"
        )
    saveData(theta0, theta1, df["km"].max(), df["km"].min())


def main():
    """Main function that run the program for training the linear
    regression model."""

    df = load_csv("data.csv")

    df = norm_value(df)

    print(df)

    learning(df)


if __name__ == "__main__":
    main()
