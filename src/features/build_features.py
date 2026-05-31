import pandas as pd
import numpy as np


def build_features(df):

    df = df.copy()

    df["hour"] = pd.to_datetime(df["ts"]).dt.hour
    df["day"] = pd.to_datetime(df["ts"]).dt.dayofweek

    # velocity features
    df = df.sort_values("ts")

    df["cust_tx_count"] = df.groupby("customer_id").cumcount()
    df["merchant_tx_count"] = df.groupby("merchant_id").cumcount()

    df["log_amount"] = np.log1p(df["amount"])

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/transactions.csv")
    df = build_features(df)
    df.to_csv("data/features.csv", index=False)