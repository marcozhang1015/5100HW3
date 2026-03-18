from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN_DIRS = [
    "pg_cartpole_rtg_baseline_CartPole-v1_17-03-2026_18-35-49",
    "pg_cartpole_na_rtg_baseline_CartPole-v1_17-03-2026_18-43-05",
]


def main():
    plt.figure(figsize=(7, 5))
    for name in RUN_DIRS:
        run_dir = Path("data") / name
        event_files = sorted(run_dir.glob("events.out.tfevents.*"))
        ea = EventAccumulator(str(event_files[0]))
        ea.Reload()

        env_steps = {e.step: e.value for e in ea.Scalars("Train_EnvstepsSoFar")}
        base_loss = {e.step: e.value for e in ea.Scalars("Baseline_Loss")}

        steps = sorted(set(env_steps.keys()) & set(base_loss.keys()))
        xs = [env_steps[s] for s in steps]
        ys = [base_loss[s] for s in steps]
        plt.plot(xs, ys, label=name)
    plt.xlabel("Environment Steps")
    plt.ylabel("Loss")
    plt.title("Baseline Loss vs Environment Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig("baseline_loss_curve.png", dpi=150)


if __name__ == "__main__":
    main()
