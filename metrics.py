"""
═══════════════════════════════════════════════════════════════════════════════
METRICS
═══════════════════════════════════════════════════════════════════════════════
Reads training results and prints a complete metrics report.
═══════════════════════════════════════════════════════════════════════════════
What it shows:
  - Best epoch by mAP50
  - Best epoch by F1 score
  - Full training history (all epochs)
  - Threshold recommendations
  - Saves metrics_report.txt
  - Saves training_curves.png (4 graphs)
═══════════════════════════════════════════════════════════════════════════════
Usage:
  python3 metrics.py
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

from utilities.data_utils import (
    print_header, print_success, print_warn, print_error, print_info,
)


def find_results_csv():
    if not Path("runs").exists():
        return None
    matches = [f for f in Path("runs").rglob("results.csv")
               if "smoketest" not in str(f)]
    return str(sorted(matches, key=lambda f: f.stat().st_mtime)[-1]) if matches else None


def metrics(csv_path=None, report_name=None):
    print_header("Metrics — Full Training Report")

    if csv_path is None:
        csv_path = find_results_csv()
    if not csv_path:
        print_error("No results.csv found. Run train.py first.")
        sys.exit(1)

    if report_name is None:
        report_name = "metrics/metrics_report.txt"

    print_info(f"Reading: {csv_path}\n")

    try:
        import pandas as pd
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "-q"])
        import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Calculate F1 from Precision and Recall
    # F1 = 2 x (P x R) / (P + R)
    df["F1"] = (
        2 * (df["metrics/precision(B)"] * df["metrics/recall(B)"])
        / (df["metrics/precision(B)"] + df["metrics/recall(B)"])
    ).fillna(0)

    best_map = df.loc[df["metrics/mAP50(B)"].idxmax()]
    best_f1  = df.loc[df["F1"].idxmax()]
    last     = df.iloc[-1]

    # Build report
    lines = []
    lines.append("=" * 60)
    lines.append("  WOUND DETECTION - YOLO26m - METRICS REPORT")
    lines.append("=" * 60)
    lines.append(f"  Model         : YOLO26m")
    lines.append(f"  Dataset       : AZH Clinical Wound Dataset")
    lines.append(f"  Run           : {Path(csv_path).parent.parent.name}")
    lines.append(f"  Total Epochs  : {int(last['epoch'])}")
    lines.append("")
    lines.append("  BEST EPOCH BY mAP50")
    lines.append("  " + "-" * 50)
    lines.append(f"  Epoch         : {int(best_map['epoch'])}")
    lines.append(f"  Box Loss      : {best_map['train/box_loss']:.4f}")
    lines.append(f"  Class Loss    : {best_map['train/cls_loss']:.4f}")
    lines.append(f"  mAP50         : {best_map['metrics/mAP50(B)']:.4f}  "
                 f"({best_map['metrics/mAP50(B)']:.1%})")
    lines.append(f"  mAP50-95      : {best_map['metrics/mAP50-95(B)']:.4f}  "
                 f"({best_map['metrics/mAP50-95(B)']:.1%})")
    lines.append(f"  Precision     : {best_map['metrics/precision(B)']:.4f}")
    lines.append(f"  Recall        : {best_map['metrics/recall(B)']:.4f}")
    lines.append(f"  F1 Score      : {best_map['F1']:.4f}")
    lines.append("")
    lines.append("  BEST EPOCH BY F1")
    lines.append("  " + "-" * 50)
    lines.append(f"  Epoch         : {int(best_f1['epoch'])}")
    lines.append(f"  Precision     : {best_f1['metrics/precision(B)']:.4f}")
    lines.append(f"  Recall        : {best_f1['metrics/recall(B)']:.4f}")
    lines.append(f"  F1 Score      : {best_f1['F1']:.4f}")
    lines.append("")
    lines.append("  CONFIDENCE THRESHOLD RECOMMENDATIONS")
    lines.append("  " + "-" * 50)
    lines.append("  Conservative (high precision) : 0.50")
    lines.append("  Balanced     (best F1)        : 0.35")
    lines.append("  Sensitive    (high recall)    : 0.25  <- medical use")
    lines.append("")
    lines.append("  FULL TRAINING HISTORY")
    lines.append("  " + "-" * 50)
    lines.append(f"  {'Epoch':>5}  {'BoxLoss':>8}  {'ClsLoss':>8}  "
                 f"{'mAP50':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")

    for _, row in df.iterrows():
        lines.append(
            f"  {int(row['epoch']):>5}  "
            f"{row['train/box_loss']:>8.4f}  "
            f"{row['train/cls_loss']:>8.4f}  "
            f"{row['metrics/mAP50(B)']:>7.4f}  "
            f"{row['metrics/precision(B)']:>7.4f}  "
            f"{row['metrics/recall(B)']:>7.4f}  "
            f"{row['F1']:>7.4f}"
        )

    lines.append("=" * 60)
    report = "\n".join(lines)

    print(report)

    # Save report with run-specific name
    report_path = Path(report_name)
    report_path.write_text(report)
    print_success(f"\nReport saved -> {report_path}")

    # Save training curves with run-specific name
    curve_name = report_name.replace(".txt", "_curves.png").replace("metrics/", "metrics/")
    _plot_training_curves(df, curve_name)


METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)

def _plot_training_curves(df, save_name="metrics/training_curves.png"):
    """Generate and save 4 training curve graphs."""
    try:
        import matplotlib.pyplot as plt

        run_name = save_name.replace("metrics_", "").replace("_curves.png", "")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"YOLO26m Wound Detection - Training Curves ({run_name})",
                     fontsize=14)

        axes[0,0].plot(df["epoch"], df["metrics/mAP50(B)"],
                       color="#1D9E75", linewidth=2)
        axes[0,0].set_title("mAP50")
        axes[0,0].set_xlabel("Epoch")
        axes[0,0].set_ylabel("mAP50")
        axes[0,0].grid(alpha=0.3)

        axes[0,1].plot(df["epoch"], df["F1"],
                       color="#378ADD", linewidth=2)
        axes[0,1].set_title("F1 Score")
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].set_ylabel("F1")
        axes[0,1].grid(alpha=0.3)

        axes[1,0].plot(df["epoch"], df["train/box_loss"],
                       color="#D85A30", linewidth=2)
        axes[1,0].set_title("Box Loss (should go down)")
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Loss")
        axes[1,0].grid(alpha=0.3)

        axes[1,1].plot(df["epoch"], df["train/cls_loss"],
                       color="#7F77DD", linewidth=2)
        axes[1,1].set_title("Class Loss (should go down)")
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylabel("Loss")
        axes[1,1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_name, dpi=150, bbox_inches="tight")
        plt.close()
        print_success(f"Training curves saved -> {save_name}")

    except Exception as e:
        print_warn(f"Could not generate plots: {e}")


def save_all_metrics():
    """Save a separate metrics report for every training run."""
    if not Path("runs").exists():
        print_error("No runs folder found. Run train.py first.")
        return

    matches = [f for f in Path("runs").rglob("results.csv")
               if "smoketest" not in str(f)]

    if not matches:
        print_error("No results.csv files found.")
        return

    print_info(f"Found {len(matches)} training run(s)\n")

    for csv_file in sorted(matches, key=lambda f: f.stat().st_mtime):
        run_name = csv_file.parent.name
        metrics(
            csv_path    = str(csv_file),
            report_name = f"metrics/{run_name}.txt"
        )

    # Print comparison summary at the end
    _print_comparison_summary(matches)


def _print_comparison_summary(matches):
    """Print a side by side comparison of all runs."""
    try:
        import pandas as pd

        print("\n" + "=" * 70)
        print("  COMPARISON SUMMARY — ALL RUNS")
        print("=" * 70)
        print(f"  {'Run':<20} {'Epochs':>6} {'BestEp':>6} "
              f"{'mAP50':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print("  " + "-" * 65)

        for csv_file in sorted(matches, key=lambda f: f.stat().st_mtime):
            run_name = csv_file.parent.name
            df = pd.read_csv(str(csv_file))
            df.columns = df.columns.str.strip()
            df["F1"] = (
                2 * (df["metrics/precision(B)"] * df["metrics/recall(B)"])
                / (df["metrics/precision(B)"] + df["metrics/recall(B)"])
            ).fillna(0)

            best = df.loc[df["metrics/mAP50(B)"].idxmax()]
            total = int(df.iloc[-1]["epoch"])

            print(
                f"  {run_name:<20} {total:>6} {int(best['epoch']):>6} "
                f"{best['metrics/mAP50(B)']:>7.4f} "
                f"{best['metrics/precision(B)']:>7.4f} "
                f"{best['metrics/recall(B)']:>7.4f} "
                f"{best['F1']:>7.4f}"
            )

        print("=" * 70)
        print_success("All reports saved!")

    except Exception as e:
        print_warn(f"Could not generate comparison: {e}")


if __name__ == "__main__":
    save_all_metrics()
