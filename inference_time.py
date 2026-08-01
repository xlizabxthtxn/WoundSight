import time
from ultralytics import YOLO
from pathlib import Path

base = Path.home() / "WoundSight/runs/detect/runs"
img = str(list((Path.home() / "WoundSight/wound_dataset/test/images").glob("*.jpg"))[0])

print(f"{'Run':<28}{'Inference (ms)':<16}{'FPS'}")
print("-" * 50)
for run in sorted(base.iterdir()):
    w = run / "weights" / "best.pt"
    if not w.exists():
        continue
    m = YOLO(str(w))
    for _ in range(3):            # warm-up (critical!)
        m(img, verbose=False)
    times = []
    for _ in range(20):           # measure
        t0 = time.time()
        m(img, verbose=False)
        times.append((time.time() - t0) * 1000)
    avg = sum(times) / len(times)
    print(f"{run.name:<28}{avg:<16.1f}{1000/avg:.1f}")
