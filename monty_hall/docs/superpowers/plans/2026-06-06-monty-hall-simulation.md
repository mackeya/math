# Monty Hall Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-file Python simulation of the Monty Hall problem with a two-panel static matplotlib visualization showing win rates and convergence to theoretical probabilities.

**Architecture:** Three pure functions in `monty_hall.py` — one for aggregate simulation, one for convergence tracking, one for plotting — wired together by `main()`. Simulation is fully vectorised with numpy for performance at N=10,000.

**Tech Stack:** Python 3, numpy, matplotlib

---

## File Structure

- Create: `monty_hall.py` — simulation logic and plotting
- Create: `tests/test_monty_hall.py` — unit tests for simulation functions

---

### Task 1: Project skeleton and failing tests

**Files:**
- Create: `tests/test_monty_hall.py`

- [ ] **Step 1: Create the test file**

```python
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
```

- [ ] **Step 2: Run the tests to confirm they fail (monty_hall.py doesn't exist yet)**

```bash
python3 -m pytest tests/test_monty_hall.py -v
```

Expected: `ModuleNotFoundError: No module named 'monty_hall'`

---

### Task 2: Implement `simulate_monty_hall`

**Files:**
- Create: `monty_hall.py`

- [ ] **Step 1: Create monty_hall.py with the simulate function**

The key insight: staying wins iff the original pick was the car (prob 1/3). Switching wins iff the original pick was NOT the car (always the complement). The host's specific door choice is irrelevant to the outcome distribution, so we skip simulating it explicitly.

```python
"""Monty Hall problem numerical simulation with matplotlib visualization."""

import numpy as np
import matplotlib.pyplot as plt


def simulate_monty_hall(n_trials: int) -> tuple[int, int]:
    """Run n_trials games and return (stay_wins, switch_wins).

    Staying wins iff the original pick equals the car door.
    Switching always wins the complement — no need to simulate host choice.
    """
    rng = np.random.default_rng()
    car = rng.integers(0, 3, size=n_trials)
    pick = rng.integers(0, 3, size=n_trials)
    stay_wins = int(np.sum(pick == car))
    switch_wins = n_trials - stay_wins
    return stay_wins, switch_wins
```

- [ ] **Step 2: Run the simulate tests**

```bash
python3 -m pytest tests/test_monty_hall.py -v -k "simulate"
```

Expected: 3 tests pass (`test_simulate_returns_correct_counts`, `test_simulate_stay_rate_near_one_third`, `test_simulate_switch_rate_near_two_thirds`), 3 skip/error on missing `compute_convergence`.

- [ ] **Step 3: Commit**

```bash
git add monty_hall.py tests/test_monty_hall.py
git commit -m "feat: add simulate_monty_hall with tests"
```

---

### Task 3: Implement `compute_convergence`

**Files:**
- Modify: `monty_hall.py`

- [ ] **Step 1: Add compute_convergence to monty_hall.py (after simulate_monty_hall)**

```python
def compute_convergence(n_trials: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative win-rate arrays of length n_trials for each strategy.

    stay_curve[i] is the stay win rate after i+1 trials.
    switch_curve is always 1 - stay_curve (the complement).
    """
    rng = np.random.default_rng()
    car = rng.integers(0, 3, size=n_trials)
    pick = rng.integers(0, 3, size=n_trials)
    stay_results = (pick == car).astype(float)
    counts = np.arange(1, n_trials + 1)
    stay_curve = np.cumsum(stay_results) / counts
    switch_curve = 1.0 - stay_curve
    return stay_curve, switch_curve
```

- [ ] **Step 2: Run all tests**

```bash
python3 -m pytest tests/test_monty_hall.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add monty_hall.py
git commit -m "feat: add compute_convergence with tests"
```

---

### Task 4: Implement `plot_results` and `main`

**Files:**
- Modify: `monty_hall.py`

- [ ] **Step 1: Add plot_results and main to monty_hall.py (after compute_convergence)**

```python
def plot_results(
    stay_rate: float,
    switch_rate: float,
    stay_curve: np.ndarray,
    switch_curve: np.ndarray,
    n_trials: int,
) -> None:
    """Render two-panel figure: bar chart of win rates + convergence plot.

    Left: simulated win rates vs theoretical values (dashed lines).
    Right: cumulative win rates over trials on a log x-axis.
    """
    fig, (ax_bar, ax_conv) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Monty Hall Problem Simulation", fontsize=14)

    # --- Left: bar chart ---
    strategies = ["Stay", "Switch"]
    rates = [stay_rate, switch_rate]
    colors = ["#4c72b0", "#dd8452"]
    bars = ax_bar.bar(strategies, rates, color=colors, width=0.5, zorder=2)
    ax_bar.axhline(1 / 3, color="#4c72b0", linestyle="--", linewidth=1.2,
                   label="Theoretical (1/3)", zorder=1)
    ax_bar.axhline(2 / 3, color="#dd8452", linestyle="--", linewidth=1.2,
                   label="Theoretical (2/3)", zorder=1)
    for bar, rate in zip(bars, rates):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.01,
            f"{rate:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Win Rate")
    ax_bar.set_title(f"Win Rate by Strategy (N = {n_trials:,})")
    ax_bar.legend()
    ax_bar.grid(axis="y", alpha=0.3)

    # --- Right: convergence plot ---
    trials = np.arange(1, n_trials + 1)
    ax_conv.plot(trials, stay_curve, color="#4c72b0", linewidth=1.2, label="Stay")
    ax_conv.plot(trials, switch_curve, color="#dd8452", linewidth=1.2, label="Switch")
    ax_conv.axhline(1 / 3, color="#4c72b0", linestyle="--", linewidth=1.0,
                    label="Theoretical (1/3)")
    ax_conv.axhline(2 / 3, color="#dd8452", linestyle="--", linewidth=1.0,
                    label="Theoretical (2/3)")
    ax_conv.set_xscale("log")
    ax_conv.set_ylim(0, 1)
    ax_conv.set_xlabel("Number of Trials")
    ax_conv.set_ylabel("Cumulative Win Rate")
    ax_conv.set_title("Convergence to Theoretical Probability")
    ax_conv.legend()
    ax_conv.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run simulation and display results."""
    N_TRIALS = 10_000
    stay_wins, switch_wins = simulate_monty_hall(N_TRIALS)
    stay_curve, switch_curve = compute_convergence(N_TRIALS)
    plot_results(
        stay_rate=stay_wins / N_TRIALS,
        switch_rate=switch_wins / N_TRIALS,
        stay_curve=stay_curve,
        switch_curve=switch_curve,
        n_trials=N_TRIALS,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests to confirm nothing broke**

```bash
python3 -m pytest tests/test_monty_hall.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 3: Smoke-test the script visually**

```bash
python3 monty_hall.py
```

Expected: a matplotlib window opens with two panels — a bar chart showing Stay ~0.333 and Switch ~0.667 with dashed theoretical lines, and a convergence plot showing both lines settling to their theoretical values.

- [ ] **Step 4: Commit**

```bash
git add monty_hall.py
git commit -m "feat: add plot_results and main, simulation complete"
```
