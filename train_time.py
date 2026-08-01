import pandas as pd
from pathlib import Path

base = Path.home() / "WoundSight/runs/detect/runs"
for run in sorted(base.iterdir()):
    csv = run / "results.csv"
    if not csv.exists():
        continue
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    if "time" in df.columns:
        secs = df["time"].iloc[-1]
        print(f"{run.name:<28} {len(df):>4} epochs   {secs/60:6.1f} min   ({secs/len(df):.1f}s/epoch)")
    else:
        print(f"{run.name:<28} no time column")
