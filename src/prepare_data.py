
from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "day.csv"

FEATURES = [
    "season",
    "mnth",
    "weekday",
    "holiday",
    "workingday",
    "yesterday_cnt",
    "last_week_cnt"
]


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Sort chronologically
    df["dteday"] = pd.to_datetime(df["dteday"])
    df = df.sort_values("dteday").reset_index(drop=True)

    # Lag features
    df["yesterday_cnt"] = df["cnt"].shift(1)
    df["last_week_cnt"] = df["cnt"].shift(7)

    # Remove rows without sufficient history
    df = df.dropna(subset=FEATURES + ["cnt"])

    # Time-based split
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    y_train = train_df["cnt"]
    y_test = test_df["cnt"]

    return X_train, X_test, y_train, y_test, train_df, test_df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, train_df, test_df = load_data()

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("Train period:",
          train_df["dteday"].min().date(),
          "to",
          train_df["dteday"].max().date())

    print("Test period:",
          test_df["dteday"].min().date(),
          "to",
          test_df["dteday"].max().date())
