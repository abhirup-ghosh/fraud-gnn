import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

N_CUSTOMERS = 5000
N_MERCHANTS = 1000
N_DEVICES = 8000
N_TX = 200000


def random_timestamp():
    start = datetime.now() - timedelta(days=90)
    return start + timedelta(seconds=random.randint(0, 90*24*3600))


def generate():

    customers = range(N_CUSTOMERS)
    merchants = range(N_MERCHANTS)
    devices = range(N_DEVICES)

    rows = []

    for i in range(N_TX):

        c = random.choice(customers)
        m = random.choice(merchants)
        d = random.choice(devices)

        amount = np.random.lognormal(4, 1)

        # fraud logic (graph-dependent signal injected later)
        fraud = 0

        if random.random() < 0.006:

            amount *= random.uniform(3, 8)

            fraud = 1

        rows.append([
            i,
            random_timestamp(),
            c,
            m,
            d,
            amount,
            fraud
        ])

    df = pd.DataFrame(rows, columns=[
        "tx_id", "ts", "customer_id",
        "merchant_id", "device_id",
        "amount", "is_fraud"
    ])

    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/transactions.csv", index=False)
    print(df.head())