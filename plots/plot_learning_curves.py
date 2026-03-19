"""Plot training returns for small and large batch experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SMALL_RUNS = [
    "pg_cartpole_CartPole-v1_17-03-2026_03-55-54",
    "pg_cartpole_rtg_CartPole-v1_17-03-2026_03-58-10",
    "pg_cartpole_na_CartPole-v1_17-03-2026_02-27-26",
    "pg_cartpole_rtg_na_CartPole-v1_17-03-2026_02-28-53",
]

LARGE_RUNS = [
    "pg_cartpole_CartPole-v1_17-03-2026_04-30-36",
    "pg_cartpole_rtg_CartPole-v1_17-03-2026_02-51-22",
    "pg_cartpole_na_CartPole-v1_17-03-2026_02-53-44",
    "pg_cartpole_rtg_na_CartPole-v1_17-03-2026_02-55-20",
]


def load_training_curve(run_dir: Path):
    event_files =sorted(run_dir.glob("events.out.tfevents.*"))
    ea = EventAccumulator(str(event_files[0]))
    ea.Reload()

    env_steps= {e.step: e.value for e in ea.Scalars("Train_EnvstepsSoFar")}
    avg_returns= {e.step: e.value for e in ea.Scalars("Train_AverageReturn")}

    shared_steps = sorted(set(env_steps.keys()) & set(avg_returns.keys()))
    xs= [env_steps[s] for s in shared_steps]
    ys= [avg_returns[s] for s in shared_steps]
    return xs, ys


def plot_group(runs, title, output_path: Path):
    plt.figure(figsize=(7, 5))
    for name, xs, ys in runs:
        plt.plot(xs, ys, label=name)
    plt.title(title)
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def main():
    small_runs =[(name, *load_training_curve(DATA_DIR / name)) for name in SMALL_RUNS]
    large_runs =[(name, *load_training_curve(DATA_DIR / name)) for name in LARGE_RUNS]

    plot_group(
        small_runs,
        "Small Batch (batch size = 1000)",
        Path("learning_curve_small_batch.png"),
    )

    plot_group(
        large_runs,
        "Large Batch (batch size = 4000)",
        Path("learning_curve_large_batch.png"),
    )


if __name__ == "__main__":
    main()
