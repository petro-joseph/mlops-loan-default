from gate import MIN_AUC, TOLERANCE, gate_problems


def test_better_model_passes():
    assert gate_problems(0.82, 0.807) == []


def test_small_dip_within_tolerance_passes():
    assert gate_problems(0.807 - TOLERANCE / 2, 0.807) == []


def test_regression_beyond_tolerance_is_blocked():
    problems = gate_problems(0.782, 0.807)
    assert len(problems) == 1 and "worse than production" in problems[0]


def test_below_minimum_is_blocked_even_without_a_current_model():
    problems = gate_problems(MIN_AUC - 0.01)
    assert len(problems) == 1 and "minimum" in problems[0]


def test_first_model_above_minimum_passes():
    assert gate_problems(0.75) == []
