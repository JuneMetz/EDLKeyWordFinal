"""
Try many training settings and compare them fairly (every setting runs SEEDS times, mean +/- std).

Uses train.py's load() and train_once(). Picks are made on CLEAN VALIDATION 3 s macro accuracy (official, see train.py); test is shown
for the record. Results go to <KWS Capstone>/local_ml/runs/sweep_<time>.csv (outside the repo), one row per
setting, written as soon as the setting finishes.

Run:
  python ML/Training/sweep.py
  python ML/Training/sweep.py --seeds 5 --epochs 80
"""
import argparse
import csv
import time

import numpy as np
import torch

from train import FEATURES, RUNS, load, train_once

# (channels, first kernel, dropout, specaug, mixup)
GRID = []
for channels in (24, 32, 48):
    for kernel in ((3, 3), (10, 4)):
        for aug in ((False, False), (True, True)):
            GRID.append((channels, kernel, 0.2) + aug)
GRID += [(32, (3, 3), 0.2, True, False), (32, (3, 3), 0.2, False, True),     # each augmentation alone
         (32, (3, 3), 0.4, True, True), (32, (10, 4), 0.4, True, True)]      # more dropout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, names, _, _ = load(FEATURES, device)
    out = RUNS / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["channels", "kernel", "dropout", "specaug", "mixup", "params", "ram_kb",
            "val3c_mean", "val3c_std", "test3c_mean", "test3c_std", "test3_mean", "test3_std", "test1_mean", "test1_std", "best_epochs"]
    cols += [f"test3c_{n}" for n in names]
    with open(out, "w", newline="") as f:
        csv.writer(f).writerow(cols)

    t0 = time.time()
    for i, (channels, kernel, dropout, specaug, mixup) in enumerate(GRID, 1):
        cfg = {"channels": channels, "kernel": kernel, "dropout": dropout,
               "specaug": specaug, "mixup": mixup, "epochs": args.epochs}
        runs = [train_once(data, len(names), cfg, seed, device) for seed in range(args.seeds)]
        m = {k: (np.mean([r[k] for r in runs]), np.std([r[k] for r in runs])) for k in ("val3c", "test3c", "test3", "test1")}
        pc = np.mean([r["test3c_per_class"] for r in runs], axis=0)
        row = [channels, f"{kernel[0]}x{kernel[1]}", dropout, specaug, mixup, runs[0]["params"],
               round(runs[0]["ram_bytes"] / 1024, 1),
               *(round(v, 4) for k in ("val3c", "test3c", "test3", "test1") for v in m[k]),
               "/".join(str(r["best_epoch"]) for r in runs), *(round(float(a), 3) for a in pc)]
        with open(out, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[{i:2d}/{len(GRID)}] ch {channels:2d} k {kernel[0]}x{kernel[1]} do {dropout} spec {specaug:d} "
              f"mix {mixup:d} | val3c {m['val3c'][0]:.3f}+/-{m['val3c'][1]:.3f}  test3c {m['test3c'][0]:.3f}  test3 {m['test3'][0]:.3f}  "
              f"test1 {m['test1'][0]:.3f} | {runs[0]['ram_bytes'] / 1024:.1f} KB | {time.time() - t0:.0f} s")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
