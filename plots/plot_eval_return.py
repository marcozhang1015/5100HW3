"""Plot evaluation returns for the four RTG/NA baseline comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

RUN_DIRS = [
    "pg_cartpole_rtg_no_baseline_CartPole-v1_17-03-2026_18-34-51",
    "pg_cartpole_rtg_baseline_CartPole-v1_17-03-2026_18-35-49",
    "pg_cartpole_na_rtg_no_baseline_CartPole-v1_17-03-2026_18-38-21",
    "pg_cartpole_na_rtg_baseline_CartPole-v1_17-03-2026_18-43-05",
]


def load_aligned_series(run_dir: Path, tag: str):
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    ea= EventAccumulator(str(event_files[0]))
    ea.Reload()

    env_steps= {e.step: e.value for e in ea.Scalars("Train_EnvstepsSoFar")}
    series= {e.step: e.value for e in ea.Scalars(tag)}

    shared_steps = sorted(set(env_steps.keys()) & set(series.keys()))
    xs =[env_steps[s] for s in shared_steps]
    ys= [series[s] for s in shared_steps]
    return xs, ys


def main():
    plt.figure(figsize=(7, 5))
    for name in RUN_DIRS:
        xs, ys = load_aligned_series(DATA_DIR / name, "Eval_AverageReturn")
        plt.plot(xs, ys, label=name)

    plt.xlabel("Environment Steps")
    plt.ylabel("Average Return")
    plt.title("Eval Return vs Environment Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eval_return_curve.png", dpi=150)


if __name__ == "__main__":
    main()
