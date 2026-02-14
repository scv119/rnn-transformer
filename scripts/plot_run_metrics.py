#!/usr/bin/env python3
import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DICT_RE = re.compile(r"\{[^{}]*\}")
STEP_RE = re.compile(r"(\d+)/(\d+)")


def parse_run_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run must use format label=path")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty")
    return label, path


def maybe_float(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def load_from_trainer_state(run_dir: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train_points: dict[int, float] = {}
    eval_points: dict[int, float] = {}

    for state_path in sorted(run_dir.glob("checkpoint-*/trainer_state.json")):
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        for entry in state.get("log_history", []):
            step = entry.get("step")
            if not isinstance(step, int):
                continue
            loss = maybe_float(entry.get("loss"))
            eval_loss = maybe_float(entry.get("eval_loss"))
            if loss is not None:
                train_points[step] = loss
            if eval_loss is not None:
                eval_points[step] = eval_loss

    train = sorted(train_points.items())
    evals = sorted(eval_points.items())
    return train, evals


def load_from_log(
    log_path: Path,
    default_train_interval: int,
    default_eval_interval: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    text = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    train_points: list[tuple[int, float]] = []
    eval_points: list[tuple[int, float]] = []
    train_fallback_step = 0
    eval_fallback_step = 0

    for raw_line in text.splitlines():
        line = ANSI_ESCAPE_RE.sub("", raw_line)
        matches = list(DICT_RE.finditer(line))
        if not matches:
            continue
        for m in matches:
            payload = m.group(0)
            try:
                entry = ast.literal_eval(payload)
            except Exception:
                continue
            if not isinstance(entry, dict):
                continue

            prefix = line[: m.start()]
            step = None
            step_hits = STEP_RE.findall(prefix)
            if step_hits:
                step = int(step_hits[-1][0])
            elif isinstance(entry.get("step"), int):
                step = int(entry["step"])

            loss = maybe_float(entry.get("loss"))
            eval_loss = maybe_float(entry.get("eval_loss"))

            if loss is not None:
                if step is None:
                    train_fallback_step += default_train_interval
                    step = train_fallback_step
                train_points.append((step, loss))

            if eval_loss is not None:
                if step is None:
                    eval_fallback_step += default_eval_interval
                    step = eval_fallback_step
                eval_points.append((step, eval_loss))

    return train_points, eval_points


def load_run(
    source: Path,
    default_train_interval: int,
    default_eval_interval: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    if source.is_dir():
        return load_from_trainer_state(source)
    if source.is_file():
        return load_from_log(source, default_train_interval, default_eval_interval)
    raise FileNotFoundError(f"Source not found: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/eval loss across multiple runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_arg,
        help="Run spec in format label=path (path can be a log file or run dir with checkpoints).",
    )
    parser.add_argument("--output", type=Path, default=Path("assets/run_metrics.png"))
    parser.add_argument("--title", type=str, default="Training Curves Comparison")
    parser.add_argument("--train-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--max-step", type=int, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_train, ax_eval) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for label, src in args.run:
        train_points, eval_points = load_run(src, args.train_interval, args.eval_interval)

        if args.max_step is not None:
            train_points = [p for p in train_points if p[0] <= args.max_step]
            eval_points = [p for p in eval_points if p[0] <= args.max_step]

        if train_points:
            xs, ys = zip(*train_points)
            ax_train.plot(xs, ys, label=label, linewidth=1.8)
        if eval_points:
            xs, ys = zip(*eval_points)
            ax_eval.plot(xs, ys, label=label, marker="o", linewidth=1.8)

    ax_train.set_ylabel("Train Loss")
    ax_eval.set_ylabel("Eval Loss")
    ax_eval.set_xlabel("Step")
    ax_train.grid(alpha=0.3)
    ax_eval.grid(alpha=0.3)
    ax_train.legend(loc="best")
    ax_eval.legend(loc="best")
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
