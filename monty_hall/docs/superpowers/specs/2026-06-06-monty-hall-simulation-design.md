# Monty Hall Simulation — Design Spec

## Overview

A single self-contained Python script (`monty_hall.py`) that numerically simulates the Monty Hall problem and produces a two-panel static matplotlib figure demonstrating both the aggregate win rates and their convergence to theoretical values.

## Architecture

One file, three pure functions, one entry point:

```
monty_hall.py
├── simulate_monty_hall(n_trials) → (stay_wins, switch_wins)
├── compute_convergence(n_trials) → (stay_curve, switch_curve)
├── plot_results(stay_rate, switch_rate, stay_curve, switch_curve, n_trials)
└── main()
```

No external dependencies beyond `numpy` and `matplotlib`.

## Simulation Logic

### `simulate_monty_hall(n_trials: int) -> tuple[int, int]`

Runs `n_trials` independent games. For each trial:

1. Place car behind a random door (1–3).
2. Contestant picks a random door.
3. Host opens one of the remaining doors — must not be the car door, must not be the contestant's door. If two doors are valid choices for the host, pick randomly among them.
4. Record: staying wins if contestant's original pick == car door. Switching wins otherwise (always the complement).

Returns `(stay_wins, switch_wins)` as integers.

### `compute_convergence(n_trials: int) -> tuple[np.ndarray, np.ndarray]`

Runs `n_trials` games and records cumulative win rates after every trial. Returns two arrays of shape `(n_trials,)`: `stay_curve` and `switch_curve`, where index `i` holds the win rate after `i+1` trials.

Implemented as a single vectorised pass using numpy random operations for performance.

## Visualization

### `plot_results(...)`

Two-panel matplotlib figure, `figsize=(12, 5)`:

**Left panel — Bar chart:**
- Two bars: "Stay" and "Switch", heights = simulated win rates
- Horizontal dashed lines at `1/3` and `2/3`, labelled "Theoretical (1/3)" and "Theoretical (2/3)"
- Win rate printed above each bar (e.g. `0.334`)
- Y-axis range [0, 1], labelled "Win Rate"
- Title: "Win Rate by Strategy (N = 10,000)"

**Right panel — Convergence plot:**
- Two lines: Stay and Switch cumulative win rates over trials 1–N
- Same horizontal dashed lines at 1/3 and 2/3
- X-axis log-scaled so early high-variance region is visible
- X-axis label: "Number of Trials", Y-axis label: "Cumulative Win Rate"
- Title: "Convergence to Theoretical Probability"
- Legend identifying Stay and Switch lines

### `main()`

Sets `N_TRIALS = 10_000`, calls `simulate_monty_hall` and `compute_convergence`, passes results to `plot_results`, calls `plt.show()`.

## Style

- Comments and docstrings on every function (concise, informative — following project CLAUDE.md convention from sibling `fluid` project)
- Pure functions; no global mutable state
- `python3` compatible

## Non-goals

- No interactivity / animation
- No GUI beyond matplotlib's default window
- No CLI argument parsing
