"""Compare default vs decreased baseline settings on loss and eval return."""

from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

BASELINE_RUNS = [
    "pg_cartpole_rtg_baseline_CartPole-v1_17-03-2026_18-35-49",
    "pg_cartpole_na_rtg_baseline_CartPole-v1_17-03-2026_18-43-05",
]

DECREASED_RUNS = [
    "pg_cartpole_rtg_baseline_decreased_CartPole-v1_17-03-2026_20-11-25",
    "pg_cartpole_na_rtg_baseline_decreased_CartPole-v1_17-03-2026_20-12-26",
]


def load_aligned_series(run_dir: Path):
    event_files =sorted(run_dir.glob("events.out.tfevents.*"))
    ea = EventAccumulator(str(event_files[0]))
    ea.Reload()

    env_steps ={e.step: e.value for e in ea.Scalars("Train_EnvstepsSoFar")}
    baseline_loss ={e.step: e.value for e in ea.Scalars("Baseline_Loss")}
    eval_return ={e.step: e.value for e in ea.Scalars("Eval_AverageReturn")}

    def align(series):
        shared_steps = sorted(set(env_steps.keys()) & set(series.keys()))
        xs = [env_steps[s] for s in shared_steps]
        ys = [series[s] for s in shared_steps]
        return xs, ys

    return align(baseline_loss), align(eval_return)


def plot_group(runs, title, ylabel, filename):
    plt.figure(figsize=(7, 5))
    for name, xs, ys in runs:
        plt.plot(xs, ys, label=name)
    plt.xlabel("Environment Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)


def main():
    base_loss_runs= []
    base_perf_runs= []
    for name in BASELINE_RUNS:
        (xs1, ys1), (xs2, ys2) = load_aligned_series(DATA_DIR / name)
        base_loss_runs.append((name, xs1, ys1))
        base_perf_runs.append((name, xs2, ys2))

    dec_loss_runs= []
    dec_perf_runs= []
    for name in DECREASED_RUNS:
        (xs1, ys1), (xs2, ys2) = load_aligned_series(DATA_DIR / name)
        dec_loss_runs.append((name, xs1, ys1))
        dec_perf_runs.append((name, xs2, ys2))

    plot_group(
        base_loss_runs + dec_loss_runs,
        "Baseline Loss (Default vs Decreased)",
        "Baseline Loss",
        "baseline_loss_compare.png",
    )

    plot_group(
        base_perf_runs + dec_perf_runs,
        "Eval Return (Default vs Decreased)",
        "Average Return",
        "eval_return_compare.png",
    )


if __name__ == "__main__":
    main()
