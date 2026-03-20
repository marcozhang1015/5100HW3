from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

data_dir = Path(__file__).resolve().parents[1] / "data"
run_dirs = [
    "pg_cartpole_rtg_baseline_CartPole-v1_17-03-2026_18-35-49",
    "pg_cartpole_na_rtg_baseline_CartPole-v1_17-03-2026_18-43-05",
]
if __name__ == "__main__":
    plt.figure(figsize=(7,5))

    for name in run_dirs:
        run_dir= data_dir / name
        event_files=sorted(run_dir.glob("events.out.tfevents.*"))
        ea= EventAccumulator(str(event_files[0]))
        ea.Reload()

        env_steps= ea.Scalars("Train_EnvstepsSoFar")
        baseline_loss= ea.Scalars("Baseline_Loss")

        d={}
        for item in env_steps:
            d[item.step]=item.value

        xs = []
        ys = []
        for item in baseline_loss:
            if item.step in d:
                xs.append(d[item.step])
                ys.append(item.value)

        plt.plot(xs,ys,label=name)

    plt.xlabel("Environment Steps")
    plt.ylabel("Loss")
    plt.title("Baseline Loss vs Environment Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig("baseline_loss_curve.png",dpi=150)
