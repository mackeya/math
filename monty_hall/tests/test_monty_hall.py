# tests/test_monty_hall.py
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from monty_hall import simulate_monty_hall, compute_convergence


def test_simulate_returns_correct_counts():
    """stay_wins + switch_wins == n_trials always."""
    stay, switch = simulate_monty_hall(1000)
    assert stay + switch == 1000


def test_simulate_stay_rate_near_one_third():
    """Stay win rate should converge to ~1/3 for large N."""
    stay, switch = simulate_monty_hall(50_000)
    stay_rate = stay / 50_000
    assert abs(stay_rate - 1 / 3) < 0.02


def test_simulate_switch_rate_near_two_thirds():
    """Switch win rate should converge to ~2/3 for large N."""
    stay, switch = simulate_monty_hall(50_000)
    switch_rate = switch / 50_000
    assert abs(switch_rate - 2 / 3) < 0.02


def test_convergence_array_lengths():
    """Both convergence arrays should have length n_trials."""
    n = 500
    stay_curve, switch_curve = compute_convergence(n)
    assert len(stay_curve) == n
    assert len(switch_curve) == n


def test_convergence_curves_sum_to_one():
    """Stay and switch curves are complements — sum to 1 at every point."""
    stay_curve, switch_curve = compute_convergence(500)
    np.testing.assert_allclose(stay_curve + switch_curve, 1.0)


def test_convergence_final_values_near_theory():
    """Final values of convergence arrays should be near 1/3 and 2/3."""
    stay_curve, switch_curve = compute_convergence(50_000)
    assert abs(stay_curve[-1] - 1 / 3) < 0.02
    assert abs(switch_curve[-1] - 2 / 3) < 0.02
