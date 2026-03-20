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
def main():
    plt.figure(figsize=(7,5))

    for name in RUN_DIRS:
        run_dir= DATA_DIR / name
        event_files=sorted(run_dir.glob("events.out.tfevents.*"))
        ea= EventAccumulator(str(event_files[0]))
        ea.Reload()

        env_steps= ea.Scalars("Train_EnvstepsSoFar")
        eval_returns= ea.Scalars("Eval_AverageReturn")

        step_to_env = {}
        for item in env_steps:
            step_to_env[item.step]= item.value

        xs=[]
        ys=[]
        for item in eval_returns:
            if item.step in step_to_env:
                xs.append(step_to_env[item.step])
                ys.append(item.value)

        plt.plot(xs,ys,label=name)

    plt.xlabel("Environment Steps")
    plt.ylabel("Average Return")
    plt.title("Eval Return vs Environment Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eval_return_curve.png",dpi=150)


if __name__ == "__main__":
    main()
