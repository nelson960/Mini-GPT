from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List


CONFIG_MARKER = "--- Configuration ---"
DEFAULT_COLUMNS = [
    "run",
    "best_val",
    "best_step",
    "total_steps",
    "n_layer",
    "n_head",
    "n_embd",
    "dropout",
    "use_moe",
    "n_experts",
    "moe_top_k",
    "moe_aux_loss_coef",
    "ffn_mult",
    "lr",
    "weight_decay",
    "eval_iters",
    "patience",
    "min_delta",
]


def parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw == "None":
        return None
    if raw == "True":
        return True
    if raw == "False":
        return False

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        return raw


def parse_results(path: Path) -> Dict[str, Any]:
    section = "metrics"
    metrics: Dict[str, Any] = {}
    cfg: Dict[str, Any] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line == CONFIG_MARKER:
            section = "config"
            continue
        if "=" not in line:
            continue

        key, raw = [x.strip() for x in line.split("=", 1)]
        val = parse_value(raw)
        if section == "metrics":
            metrics[key] = val
        else:
            cfg[key] = val

    out: Dict[str, Any] = {}
    out.update(metrics)
    out.update(cfg)
    return out


def format_cell(v: Any, key: str) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if key in {"best_val", "final_train_loss", "final_val_loss"}:
            return f"{v:.4f}"
        if abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.6g}"
    return str(v)


def collect(runs_dir: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        res = run_dir / "results.txt"
        if not res.exists():
            continue
        row = parse_results(res)
        row["run"] = run_dir.name
        recs.append(row)
    return recs


def sort_rows(rows: List[Dict[str, Any]], sort_by: str, descending: bool) -> List[Dict[str, Any]]:
    def key_fn(r: Dict[str, Any]) -> Any:
        v = r.get(sort_by)
        if isinstance(v, (int, float)):
            return (0, float(v))
        return (1, float("inf"))

    return sorted(rows, key=key_fn, reverse=descending)


def write_markdown(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    lines: List[str] = ["# Run Leaderboard", ""]
    if not rows:
        lines.append("No completed runs found.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    header = ["rank"] + columns
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for i, r in enumerate(rows, start=1):
        row = [str(i)]
        for c in columns:
            row.append(format_cell(r.get(c), c))
        lines.append("| " + " | ".join(row) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in columns})


def main() -> None:
    p = argparse.ArgumentParser(description="Build runs leaderboard from runs/*/results.txt")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--output_md", default="runs/leaderboard.md")
    p.add_argument("--output_csv", default="runs/leaderboard.csv")
    p.add_argument("--sort_by", default="best_val")
    p.add_argument("--descending", action="store_true")
    p.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = collect(runs_dir)
    rows = sort_rows(rows, sort_by=args.sort_by, descending=args.descending)

    md_path = Path(args.output_md)
    csv_path = Path(args.output_csv)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_markdown(md_path, rows, args.columns)
    write_csv(csv_path, rows, args.columns)

    print(f"wrote: {md_path}")
    print(f"wrote: {csv_path}")
    if rows:
        top = rows[0]
        print(f"top run: {top.get('run')} | {args.sort_by}={top.get(args.sort_by)}")


if __name__ == "__main__":
    main()
