"""Population Stability Index (PSI): how far recent inputs have moved from training.

At training time each feature gets a profile: the share of training rows in
each bin (numbers, cut at the deciles) or in each category. In serving, the
same shares are computed over the recent window and compared:

    PSI = sum over bins of (recent% - training%) * ln(recent% / training%)

Rule of thumb used in credit risk: < 0.1 stable, 0.1-0.25 watch, > 0.25 shifted.
"""
import numpy as np
import pandas as pd

EPS = 1e-4
OTHER = "__other__"
SCORE_EDGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _num_props(values, edges):
    counts = np.histogram(np.asarray(values, dtype=float), bins=[-np.inf, *edges, np.inf])[0]
    return (counts / max(len(values), 1)).tolist()


def profile(series, edges=None):
    """Training-time profile of one column (or of the model's scores)."""
    if edges is None and (isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object):
        return {"type": "cat", "props": series.astype(str).value_counts(normalize=True).to_dict()}
    if edges is None:
        edges = sorted(set(np.quantile(series, np.linspace(0.1, 0.9, 9)).tolist()))
    return {"type": "num", "edges": edges, "props": _num_props(series, edges)}


def psi(prof, values):
    """PSI of recent `values` against a training profile."""
    if prof["type"] == "num":
        ref, cur = prof["props"], _num_props(values, prof["edges"])
    else:
        known = prof["props"]
        counts = pd.Series([v if v in known else OTHER for v in map(str, values)]).value_counts(normalize=True)
        keys = [*known, OTHER]
        ref = [known.get(k, 0.0) for k in keys]
        cur = [float(counts.get(k, 0.0)) for k in keys]
    r = np.clip(ref, EPS, None)
    c = np.clip(cur, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))
