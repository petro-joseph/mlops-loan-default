import pandas as pd

from drift import profile, psi


def test_same_distribution_is_stable():
    train = pd.Series(range(1000))
    assert psi(profile(train), list(range(0, 1000, 3))) < 0.05


def test_shifted_numbers_are_flagged():
    train = pd.Series(range(1000))
    assert psi(profile(train), [v * 3 for v in range(0, 1000, 3)]) > 0.25


def test_shifted_categories_are_flagged():
    train = pd.Series(["car"] * 70 + ["education"] * 30, dtype="category")
    prof = profile(train)
    assert psi(prof, ["car"] * 70 + ["education"] * 30) < 0.01
    assert psi(prof, ["car"] * 20 + ["education"] * 80) > 0.25


def test_unseen_category_counts_as_drift():
    prof = profile(pd.Series(["car", "tv"] * 50, dtype="category"))
    assert psi(prof, ["crypto"] * 100) > 0.25
