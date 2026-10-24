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
