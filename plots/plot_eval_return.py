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
        eval_ret = {e.step: e.value for e in ea.Scalars("Eval_AverageReturn")}

        steps = sorted(set(env_steps.keys()) & set(eval_ret.keys()))
        xs = [env_steps[s] for s in steps]
        ys = [eval_ret[s] for s in steps]
        plt.plot(xs, ys, label=name)
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Return")
    plt.title("Eval Return vs Environment Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eval_return_curve.png", dpi=150)


if __name__ == "__main__":
    main()
