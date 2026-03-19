import argparse
import os
from pathlib import Path


def _try_import_tensorboard():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore
        return EventAccumulator
    except Exception as exc:
        raise SystemExit(
            "tensorboard is required to read event files. "
            "Please run this script in the same Python environment where you ran training.\n"
            f"Import error: {type(exc).__name__}: {exc}"
        )


def _pick_event_file(run_dir: Path) -> Path | None:
    events = list(run_dir.glob("events.out.tfevents.*"))
    if not events:
        return None
    # pick the largest file (usually most complete)
    return max(events, key=lambda p: p.stat().st_size)


def _escape_latex(s: str) -> str:
    return s.replace("_", "\\_")


def _infer_flags(run_name: str) -> str:
    flags = []
    if "_rtg_" in f"_{run_name}_":
        flags.append("rtg")
    if "_na_" in f"_{run_name}_":
        flags.append("na")
    if "_baseline_" in f"_{run_name}_":
        flags.append("baseline")
    if "_baseline_decreased_" in f"_{run_name}_":
        flags.append("baseline-decreased")
    if not flags:
        flags.append("none")
    return ",".join(flags)


def _infer_command(run_name: str) -> str:
    env_name = "CartPole-v1"
    exp_name = run_name
    if run_name.startswith("pg_"):
        exp_name = run_name[len("pg_") :]
    if "_CartPole-v1_" in exp_name:
        exp_name = exp_name.split("_CartPole-v1_")[0]
    flags = []
    if "rtg" in exp_name.split("_"):
        flags.append("-rtg")
    if "na" in exp_name.split("_"):
        flags.append("-na")
    if "baseline" in exp_name.split("_"):
        flags.append("--use_baseline")
    cmd = f"python run.py --env_name {env_name} --exp_name {exp_name}"
    if flags:
        cmd += " " + " ".join(flags)
    return cmd


def _last_scalar(items):
    if not items:
        return None
    # items have attributes: step, value
    return max(items, key=lambda x: x.step).value


def _max_scalar(items):
    if not items:
        return None
    return max(items, key=lambda x: x.value).value


def main():
    parser = argparse.ArgumentParser(description="Summarize all runs in data/ into a LaTeX table.")
    parser.add_argument("--data_dir", default="data", help="Path to data folder.")
    parser.add_argument("--out_tex", default="tuning_table.tex", help="Output LaTeX file.")
    parser.add_argument(
        "--cmds_json",
        default=None,
        help="Optional JSON mapping of run directory name to exact command string.",
    )
    args = parser.parse_args()

    EventAccumulator = _try_import_tensorboard()

    cmds_map = {}
    if args.cmds_json:
        cmds_path = Path(args.cmds_json)
        if cmds_path.exists():
            cmds_map = json.loads(cmds_path.read_text(encoding="utf-8"))

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    rows = []
    for run_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        event_file = _pick_event_file(run_dir)
        if event_file is None:
            continue
        try:
            ea = EventAccumulator(str(event_file))
            ea.Reload()
            scalars = ea.Tags().get("scalars", [])
            eval_returns = ea.Scalars("Eval_AverageReturn") if "Eval_AverageReturn" in scalars else []
            env_steps = ea.Scalars("Train_EnvstepsSoFar") if "Train_EnvstepsSoFar" in scalars else []
            baseline_loss = ea.Scalars("Baseline_Loss") if "Baseline_Loss" in scalars else []

            row = {
                "run": run_dir.name,
                "flags": _infer_flags(run_dir.name),
                "cmd": cmds_map.get(run_dir.name, _infer_command(run_dir.name)),
                "steps": _last_scalar(env_steps) or (max(s.step for s in eval_returns) if eval_returns else None),
                "eval_last": _last_scalar(eval_returns),
                "eval_max": _max_scalar(eval_returns),
                "baseline_last": _last_scalar(baseline_loss),
            }
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "run": run_dir.name,
                    "flags": _infer_flags(run_dir.name),
                    "cmd": cmds_map.get(run_dir.name, _infer_command(run_dir.name)),
                    "steps": None,
                    "eval_last": None,
                    "eval_max": None,
                    "baseline_last": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l l l r r r r}")
    lines.append("\\hline")
    lines.append("Run & Flags & Command & Steps & Eval@End & Eval@Max & BaselineLoss@End \\\\")
    lines.append("\\hline")
    for r in rows:
        run = _escape_latex(r["run"])
        flags = _escape_latex(r["flags"])
        cmd = _escape_latex(r["cmd"])
        steps = "-" if r["steps"] is None else f"{int(r['steps'])}"
        eval_last = "-" if r["eval_last"] is None else f"{r['eval_last']:.2f}"
        eval_max = "-" if r["eval_max"] is None else f"{r['eval_max']:.2f}"
        baseline_last = "-" if r["baseline_last"] is None else f"{r['baseline_last']:.2f}"
        lines.append(
            f"{run} & {flags} & \\texttt{{{cmd}}} & {steps} & {eval_last} & {eval_max} & {baseline_last} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Training runs summary from event logs.}")
    lines.append("\\label{tab:tuning-summary}")
    lines.append("\\end{table}")

    Path(args.out_tex).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_tex} with {len(rows)} runs.")


if __name__ == "__main__":
    main()
