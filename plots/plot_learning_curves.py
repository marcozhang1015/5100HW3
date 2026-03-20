from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

data_dir = Path(__file__).resolve().parents[1] / "data"
small_runs = [
    "pg_cartpole_CartPole-v1_17-03-2026_03-55-54",
    "pg_cartpole_rtg_CartPole-v1_17-03-2026_03-58-10",
    "pg_cartpole_na_CartPole-v1_17-03-2026_02-27-26",
    "pg_cartpole_rtg_na_CartPole-v1_17-03-2026_02-28-53",
]
large_runs = [
    "pg_cartpole_CartPole-v1_17-03-2026_04-30-36",
    "pg_cartpole_rtg_CartPole-v1_17-03-2026_02-51-22",
    "pg_cartpole_na_CartPole-v1_17-03-2026_02-53-44",
    "pg_cartpole_rtg_na_CartPole-v1_17-03-2026_02-55-20",
]
if __name__ == "__main__":
    plt.figure(figsize=(7,5))
    for name in small_runs:
        run_dir= data_dir / name
        event_files = sorted(run_dir.glob("events.out.tfevents.*"))
        ea= EventAccumulator(str(event_files[0]))
        ea.Reload()

        env_steps= ea.Scalars("Train_EnvstepsSoFar")
        avg_returns = ea.Scalars("Train_AverageReturn")

        d={}
        for item in env_steps:
            d[item.step] = item.value

        xs=[]
        ys=[]
        for item in avg_returns:
            if item.step in d:
                xs.append(d[item.step])
                ys.append(item.value)

        plt.plot(xs,ys,label=name)

    plt.title("Small Batch (batch size = 1000)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path("learning_curve_small_batch.png"),dpi=150)

    plt.figure(figsize=(7,5))
    for name in large_runs:
        run_dir= data_dir / name
        event_files= sorted(run_dir.glob("events.out.tfevents.*"))
        ea = EventAccumulator(str(event_files[0]))
        ea.Reload()

        env_steps = ea.Scalars("Train_EnvstepsSoFar")
        avg_returns= ea.Scalars("Train_AverageReturn")

        d = {}
        for item in env_steps:
            d[item.step]= item.value

        xs = []
        ys = []
        for item in avg_returns:
            if item.step in d:
                xs.append(d[item.step])
                ys.append(item.value)

        plt.plot(xs,ys,label=name)

    plt.title("Large Batch (batch size = 4000)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path("learning_curve_large_batch.png"),dpi=150)
