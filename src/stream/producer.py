import time
import requests
import random


while True:

    payload = {
        "amount": random.random() * 1000,
        "hour": random.randint(0, 23),
        "day": random.randint(0, 6)
    }

    r = requests.post("http://localhost:8000/score", json=payload)

    print(r.json())

    time.sleep(0.2)