"""Promote and roll back model versions through the `production` alias.

Serving always loads `models:/loan-default@production`, so moving the alias is
the deployment and moving it back is the rollback. No files are copied.

    python src/registry.py status
    python src/registry.py promote 2            # quality gate applies
    python src/registry.py promote 2 --force    # override, recorded on the version
    python src/registry.py rollback             # back to the previous production version

Quality gate: the candidate's val_auc must be at least MIN_AUC and no more than
TOLERANCE below the current production version. A rollback skips the gate: it
returns to a version that already passed it and ran in production.
"""
import argparse
import os
import sys
from datetime import datetime, timezone

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "loan-default"
ALIAS = "production"
MIN_AUC = 0.70
TOLERANCE = 0.01
HISTORY_TAG = "production_history"  # comma-separated versions, oldest first

client = MlflowClient()


def auc(version):
    mv = client.get_model_version(MODEL_NAME, version)
    return client.get_run(mv.run_id).data.metrics["val_auc"]


def current():
    try:
        return client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
    except MlflowException:
        return None


def history():
    tag = client.get_registered_model(MODEL_NAME).tags.get(HISTORY_TAG, "")
    return [v for v in tag.split(",") if v]


def point_alias(version, reason, hist):
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    client.set_registered_model_tag(MODEL_NAME, HISTORY_TAG, ",".join(hist))
    stamp = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z by {os.environ.get('PROMOTED_BY', 'unknown')}: {reason}"
    client.set_model_version_tag(MODEL_NAME, version, "last_promotion", stamp)
    print(f"{ALIAS} -> v{version}  ({reason})")


def status(_):
    prod = current()
    print(f"{MODEL_NAME}: {ALIAS} = v{prod}" if prod else f"{MODEL_NAME}: no {ALIAS} version yet")
    print(f"history: {' -> '.join(history()) or '-'}")
    for mv in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda m: int(m.version)):
        run = client.get_run(mv.run_id)
        mark = "*" if mv.version == prod else " "
        print(f" {mark} v{mv.version}  auc={run.data.metrics['val_auc']:.3f}  "
              f"model={run.data.params.get('model')}  {mv.tags.get('last_promotion', '')}")


def promote(args):
    v, prod = str(args.version), current()
    cand = auc(v)
    problems = []
    if cand < MIN_AUC:
        problems.append(f"val_auc {cand:.3f} < minimum {MIN_AUC}")
    if prod and cand < auc(prod) - TOLERANCE:
        problems.append(f"val_auc {cand:.3f} worse than production v{prod} ({auc(prod):.3f})")
    if problems and not args.force:
        sys.exit("BLOCKED: " + "; ".join(problems) + "  (use --force to override)")
    if v == prod:
        sys.exit(f"v{v} is already {ALIAS}")
    reason = "promoted, FORCED past gate: " + "; ".join(problems) if problems else f"promoted, val_auc {cand:.3f}"
    point_alias(v, reason, history() + [v])


def rollback(_):
    hist = history()
    if len(hist) < 2:
        sys.exit("nothing to roll back to")
    point_alias(hist[-2], f"rollback from v{hist[-1]}", hist[:-1])


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=status)
    pr = sub.add_parser("promote")
    pr.add_argument("version", type=int)
    pr.add_argument("--force", action="store_true")
    pr.set_defaults(fn=promote)
    sub.add_parser("rollback").set_defaults(fn=rollback)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
