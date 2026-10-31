"""Train a loan-default model, track the run in MLflow and register it.

Every run logs its parameters, validation metrics and the exact data it saw
(row count + content hash), plus `reference.json`: the training distribution
of every feature and of the validation scores, which serving compares live
traffic against to detect drift. It then registers the model as a new version
of `loan-default`. Registering does not deploy: only `registry.py promote`
moves the `production` alias.

    python src/train.py --model logreg --C 1.0
    python src/train.py --model rf --n-estimators 300 --max-depth 8
"""
import argparse
import json
import os
import tempfile

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from drift import SCORE_EDGES, profile

MODEL_NAME = "loan-default"
EXPERIMENT = "loan-default"
OPENML_ID = 31  # German Credit: 1,000 loans, 20 applicant features, good/bad outcome
SEED = 42


def load_data():
    df = fetch_openml(data_id=OPENML_ID, as_frame=True, data_home="data").frame
    y = (df.pop("class") == "bad").astype(int)  # 1 = defaulted
    return df, y


def build(args):
    prep = ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_include="number")),
        ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include="category")),
    ])
    if args.model == "logreg":
        clf = LogisticRegression(C=args.C, max_iter=1000, class_weight="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                     class_weight="balanced", random_state=SEED)
    return Pipeline([("prep", prep), ("clf", clf)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=None)
    args = p.parse_args()

    X, y = load_data()
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        params = {"model": args.model, "seed": SEED}
        params.update({"C": args.C} if args.model == "logreg"
                      else {"n_estimators": args.n_estimators, "max_depth": args.max_depth})
        mlflow.log_params(params)
        mlflow.set_tags({
            "git_sha": os.environ.get("GIT_SHA", "unknown"),
            "data.source": f"openml:{OPENML_ID}",
            "data.rows": len(X),
            "data.hash": format(pd.util.hash_pandas_object(X).sum() & 0xFFFFFFFF, "08x"),
        })

        pipe = build(args).fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "val_auc": roc_auc_score(y_val, proba),
            "val_f1": f1_score(y_val, pred),
            "val_accuracy": accuracy_score(y_val, pred),
        }
        mlflow.log_metrics(metrics)

        reference = {
            "features": {col: profile(X_tr[col]) for col in X_tr.columns},
            "score": profile(pd.Series(proba), edges=SCORE_EDGES),
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reference.json")
            with open(path, "w") as f:
                json.dump(reference, f)
            mlflow.log_artifact(path)

        info = mlflow.sklearn.log_model(
            pipe, "model",
            signature=infer_signature(X_val, proba),
            input_example=X_val.head(3),
            registered_model_name=MODEL_NAME,
        )

    print(f"run {run.info.run_id[:8]}  {params}")
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    print(f"  registered {MODEL_NAME} v{info.registered_model_version} (not promoted)")


if __name__ == "__main__":
    main()
