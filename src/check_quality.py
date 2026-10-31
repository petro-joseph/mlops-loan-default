"""CI step: fail the build if a freshly trained candidate is worse than the baseline.

    python src/check_quality.py metrics.json [quality/baseline.json]

The baseline is the accepted model's validation AUC, kept in the repo so that
raising it is itself a reviewed change. Writes a Markdown table to
$GITHUB_STEP_SUMMARY when run in GitHub Actions.
"""
import json
import os
import sys

from gate import MIN_AUC, TOLERANCE, gate_problems


def main():
    metrics = json.load(open(sys.argv[1]))
    baseline = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "quality/baseline.json"))
    cand, base = metrics["val_auc"], baseline["val_auc"]
    problems = gate_problems(cand, base, "baseline")
    verdict = "FAIL" if problems else "PASS"

    lines = [
        f"### Model quality gate: {verdict}",
        "",
        "| | val_auc |",
        "|---|---|",
        f"| candidate ({metrics.get('params', {})}) | {cand:.3f} |",
        f"| baseline | {base:.3f} |",
        f"| allowed minimum | {max(MIN_AUC, base - TOLERANCE):.3f} |",
        "",
        *[f"- {p}" for p in problems],
    ]
    print("\n".join(lines))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("\n".join(lines) + "\n")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
