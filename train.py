""""""

import pandas as pd
import sys

def load_csv(path: str) -> pd.DataFrame | None:
    """Function that receive Dataset in csv format,
    and return it as Pandas DataFrame if none error occured."""

    try:
        if not isinstance(path, str):
            raise TypeError("path must be a string.")
        if not path.lower().endswith((".csv")):
            raise ValueError("file extension must be .csv.")

        df = pd.read_csv(path)
        return df

    except (TypeError, ValueError) as error:
        print(f"{type(error).__name__}: {error}")
        return None
    except FileNotFoundError as error:
        print(f"{type(error).__name__}: {error}")
        return None
    except PermissionError as error:
        print(f"{type(error).__name__}: {error}")
        return None


def norm_value(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize km value to value between 0, 1."""
    df["km_norm"] = (df["km"] - df["km"].min()) / (df["km"].max() - df["km"].min())
    return df


def learning(df: pd.DataFrame):
    """Train the model and perform a linear regression on the DataFrame."""
    theta0, theta1 = 0
    learningRate = 0.01
    m = len(df)


def main():
    """Main function that run the program."""

    df = load_csv("data.csv")
    if df is None:
        sys.exit(1)

    df = norm_value(df)
    print(df)
    


if __name__ == "__main__":
    main()
