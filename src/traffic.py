"""Send loan applications to the API, standing in for the bank's loan system.

Applicants are drawn at random from the dataset (answers removed). With
--drift the population changes the way it might after a new product launch:
loans three times larger, twice as long, and mostly for education.

    python src/traffic.py --seconds 120 --rps 5
    python src/traffic.py --seconds 120 --rps 5 --drift
"""
import argparse
import json
import os
import random
import time
import urllib.request

from train import load_data

API = os.environ.get("API_URL", "http://api:8000")


def shift(loan):
    loan["credit_amount"] *= 3
    loan["duration"] = min(loan["duration"] * 2, 72)
    if random.random() < 0.6:
        loan["purpose"] = "education"
    return loan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=int, default=120)
    p.add_argument("--rps", type=float, default=5)
    p.add_argument("--drift", action="store_true")
    args = p.parse_args()

    X, _ = load_data()
    loans = json.loads(X.to_json(orient="records"))
    sent, refer, errors = 0, 0, 0
    end = time.time() + args.seconds
    while time.time() < end:
        loan = dict(random.choice(loans))
        if args.drift:
            loan = shift(loan)
        req = urllib.request.Request(f"{API}/predict", data=json.dumps({"loans": [loan]}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                refer += json.load(r)["results"][0]["decision"] == "refer"
        except Exception:
            errors += 1
        sent += 1
        time.sleep(1 / args.rps)
    print(f"sent {sent} ({'drifted' if args.drift else 'normal'}): referred {refer}, errors {errors}")


if __name__ == "__main__":
    main()
