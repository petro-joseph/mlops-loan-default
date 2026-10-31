"""Promote and roll back model versions through the `staging` and `production` aliases.

Each environment's API serves `models:/loan-default@<its alias>`, so moving an
alias is the deployment and moving it back is the rollback. No files are copied:
the version tested in staging is byte-for-byte the one promoted to production.

    python src/registry.py status
    python src/registry.py promote 5 --to staging   # try it in staging first
    python src/registry.py promote 5                # production (default); quality gate applies
    python src/registry.py promote 5 --force        # override, recorded on the version
    python src/registry.py rollback                 # production back to its previous version
    python src/registry.py rollback --alias staging

Quality gate (both aliases): the candidate's val_auc must be at least MIN_AUC
and no more than TOLERANCE below the current production version. A rollback
skips the gate: it returns to a version that already passed it.
"""
import argparse
import os
import sys
from datetime import datetime, timezone

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "loan-default"
ALIASES = ["staging", "production"]
MIN_AUC = 0.70
TOLERANCE = 0.01

client = MlflowClient()


def auc(version):
    mv = client.get_model_version(MODEL_NAME, version)
    return client.get_run(mv.run_id).data.metrics["val_auc"]


def current(alias):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias).version
    except MlflowException:
        return None


def history(alias):
    """Versions the alias has pointed at, oldest first."""
    tag = client.get_registered_model(MODEL_NAME).tags.get(f"{alias}_history", "")
    return [v for v in tag.split(",") if v]


def point_alias(alias, version, reason, hist):
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    client.set_registered_model_tag(MODEL_NAME, f"{alias}_history", ",".join(hist))
    stamp = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z by {os.environ.get('PROMOTED_BY', 'unknown')}: {reason}"
    client.set_model_version_tag(MODEL_NAME, version, f"last_{alias}", stamp)
    print(f"{alias} -> v{version}  ({reason})")


def status(_):
    live = {a: current(a) for a in ALIASES}
    for a in ALIASES:
        print(f"{a:<10} = v{live[a]}   history: {' -> '.join(history(a)) or '-'}" if live[a]
              else f"{a:<10} = (none)")
    for mv in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda m: int(m.version)):
        run = client.get_run(mv.run_id)
        marks = ",".join(a for a in ALIASES if live[a] == mv.version)
        last = mv.tags.get("last_production") or mv.tags.get("last_promotion", "")
        print(f"  v{mv.version:<3} auc={run.data.metrics['val_auc']:.3f}  model={run.data.params.get('model'):<7}"
              f"{'@' + marks if marks else '':<22}{last}")


def promote(args):
    v, alias, prod = str(args.version), args.to, current("production")
    cand = auc(v)
    problems = []
    if cand < MIN_AUC:
        problems.append(f"val_auc {cand:.3f} < minimum {MIN_AUC}")
    if prod and cand < auc(prod) - TOLERANCE:
        problems.append(f"val_auc {cand:.3f} worse than production v{prod} ({auc(prod):.3f})")
    if problems and not args.force:
        sys.exit("BLOCKED: " + "; ".join(problems) + "  (use --force to override)")
    if v == current(alias):
        sys.exit(f"v{v} is already {alias}")
    reason = "promoted, FORCED past gate: " + "; ".join(problems) if problems else f"promoted, val_auc {cand:.3f}"
    point_alias(alias, v, reason, history(alias) + [v])


def rollback(args):
    hist = history(args.alias)
    if len(hist) < 2:
        sys.exit(f"nothing to roll {args.alias} back to")
    point_alias(args.alias, hist[-2], f"rollback from v{hist[-1]}", hist[:-1])


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=status)
    pr = sub.add_parser("promote")
    pr.add_argument("version", type=int)
    pr.add_argument("--to", choices=ALIASES, default="production")
    pr.add_argument("--force", action="store_true")
    pr.set_defaults(fn=promote)
    rb = sub.add_parser("rollback")
    rb.add_argument("--alias", choices=ALIASES, default="production")
    rb.set_defaults(fn=rollback)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
