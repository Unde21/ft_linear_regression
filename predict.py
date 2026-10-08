"""First program, used to predict the price of a car
    for a given mileage. This one use the model trained
    by the second program."""


import pandas as pd
import sys
import json
from pathlib import Path


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


def main():
    """Main function that run the program."""

    mileage = get_mileage()
    print(mileage)
    path = "save_data.json"
    try:
        with open(path, "r") as f:
            data = json.load(f)
            theta0 = data["theta0"]
            theta1 = data["theta1"]
            max_km = data["max_km"]
            min_km = data["min_km"]
        print(data)

    except (FileNotFoundError, PermissionError,
            json.JSONDecodeError) as error:
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)


    # Add security ! 
    path = Path(path)
    path.unlink(missing_ok=True)
    print("save_data as been removed successfully.")
    
if __name__ == "__main__":
    main()
