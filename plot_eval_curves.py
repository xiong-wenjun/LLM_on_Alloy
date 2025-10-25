import argparse
import json
import os
from typing import Dict, List, Tuple


def load_log_history(output_dir: str) -> List[Dict]:
    """
    Loads Trainer log history from trainer_state.json or trainer_log.jsonl.

    Preference order:
    1) trainer_state.json -> ["log_history"]
    2) trainer_log.jsonl -> each line is a JSON log record
    """
    state_path = os.path.join(output_dir, "trainer_state.json")
    log_path = os.path.join(output_dir, "trainer_log.jsonl")

    if os.path.isfile(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("log_history", [])

    logs: List[Dict] = []
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue
        return logs

    raise FileNotFoundError(
        f"No trainer_state.json or trainer_log.jsonl found in {output_dir}"
    )


def extract_eval_series(log_history: List[Dict]) -> Dict[str, List[Tuple[int, float]]]:
    """
    Extract evaluation metrics across steps. We consider keys that look like evaluation metrics:
    - start with 'eval_' (e.g., eval_loss, eval_runtime)
    - contain 'eval/' prefix (some trainers log namespaced metrics)

    Returns a dict: metric_name -> list of (step, value) sorted by step.
    """
    series: Dict[str, List[Tuple[int, float]]] = {}

    def is_eval_key(k: str) -> bool:
        return k.startswith("eval_") or k.startswith("eval/")

    for rec in log_history:
        step = rec.get("step")
        if step is None:
            # Some logs may not include 'step'; try 'global_step' or skip
            step = rec.get("global_step")
        if step is None:
            continue

        for k, v in rec.items():
            if not is_eval_key(k):
                continue
            # Only keep scalar numeric values
            if isinstance(v, (int, float)):
                series.setdefault(k, []).append((int(step), float(v)))

    # sort by step
    for k in list(series.keys()):
        pts = sorted(series[k], key=lambda x: x[0])
        series[k] = pts

    return series


def plot_series(series: Dict[str, List[Tuple[int, float]]], save_dir: str, show: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to plot. Install with: pip install matplotlib"
        ) from e

    os.makedirs(save_dir, exist_ok=True)

    # One figure per metric
    for metric, points in series.items():
        if not points:
            continue
        steps = [s for s, _ in points]
        values = [v for _, v in points]
        plt.figure()
        plt.plot(steps, values, marker="o", linewidth=1.5)
        plt.title(metric)
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.grid(True, linestyle=":", alpha=0.5)
        safe_name = metric.replace("/", "_")
        out_path = os.path.join(save_dir, f"{safe_name}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        if show:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot validation curves from HF/TRL Trainer logs")
    parser.add_argument("--output_dir", required=True, help="Path to training output directory containing trainer_state.json or trainer_log.jsonl")
    parser.add_argument("--save_dir", default=None, help="Directory to save plots (default: <output_dir>/plots)")
    parser.add_argument("--metrics", nargs="*", default=None, help="Specific eval metric names to plot (e.g., eval_loss eval_runtime). If omitted, plot all eval_* metrics found.")
    parser.add_argument("--show", action="store_true", help="Show figures interactively in addition to saving")
    parser.add_argument("--list-metrics", action="store_true", help="Only list available evaluation metrics and exit")

    args = parser.parse_args()

    logs = load_log_history(args.output_dir)
    all_series = extract_eval_series(logs)

    if not all_series:
        print("No evaluation metrics found in logs. Ensure training ran with do_eval=True and evaluation steps.")
        return

    if args.list_metrics:
        print("Available evaluation metrics:")
        for k in sorted(all_series.keys()):
            print(" -", k)
        return

    # Filter metrics if requested
    if args.metrics:
        selected = {k: v for k, v in all_series.items() if k in set(args.metrics)}
        missing = [m for m in args.metrics if m not in all_series]
        if missing:
            print("Warning: requested metrics not found:", ", ".join(missing))
        series = selected
    else:
        series = all_series

    save_dir = args.save_dir or os.path.join(args.output_dir, "plots")
    plot_series(series, save_dir, show=args.show)
    print(f"Saved {len(series)} plot(s) to: {save_dir}")


if __name__ == "__main__":
    main()
