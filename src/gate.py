"""The quality gate, shared by promotion (registry.py) and CI (check_quality.py).

A candidate passes if its validation AUC is at least MIN_AUC and no more than
TOLERANCE below the model it would replace. The tolerance stops noise from
blocking an otherwise-equal model; anything worse than that is a regression.
"""
MIN_AUC = 0.70
TOLERANCE = 0.01


def gate_problems(candidate_auc, current_auc=None, current_label="production"):
    """Reasons the candidate fails the gate; empty when it passes."""
    problems = []
    if candidate_auc < MIN_AUC:
        problems.append(f"val_auc {candidate_auc:.3f} < minimum {MIN_AUC}")
    if current_auc is not None and candidate_auc < current_auc - TOLERANCE:
        problems.append(f"val_auc {candidate_auc:.3f} worse than {current_label} ({current_auc:.3f})")
    return problems
