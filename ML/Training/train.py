"""
Train the DS-CNN bird classifier (ML/Model/model.py) on precomputed MFCC features.

Input: a features .npz made OUTSIDE this repo from June's MFCC code (Firmware/Features), one row per
1 s clip of ML/Dataset/manifest_1s.csv (same row order), with arrays:
  mfcc (N, 98, 13), label (N,), label_names (8,), split (N,) train/val/test, recording (N,), path (N,)
Default path: <KWS Capstone>/local_ml/features_june_1s.npz (next to this repo, not in it).

Scores reported:
  1 s macro acc  every 1 s clip on its own
  3 s macro acc  the device decision: softmax averaged over the 1 s slices cut from the same 3 s BirdNET clip
                 (manifest column source_clip). TARGET: 90 % macro on test.
  clean          the OFFICIAL score: same, but no-bird val/test clips are kept only when their label can be trusted
                 (trusted_no_bird below). A label audit (11-01) found many no-bird clips hold our birds that BirdNET
                 missed. Training still uses every clip. "full" = all val/test clips, reported alongside.
"macro" = mean of the 8 per-class accuracies, so the big no-bird class cannot hide a weak bird.

Steps:
  1. normalize every MFCC coefficient with the TRAIN mean and std (the chip must use the same numbers)
  2. train with Adam + cosine LR, cross-entropy with class weights 1/count
     augmentation on train only: time shift, optional time masks (--specaug) and mixup (--mixup)
     --wave: augment the AUDIO of each train batch instead (shift, gain, polarity, no-bird background mixed into
     bird clips, time mask), then compute June's MFCC on the GPU with the LOCAL port
     <KWS Capstone>/local_ml/june_mfcc/mfcc_torch.py (her recipe stays out of this repo). Val/test use the features file.
  3. keep the epoch with the best CLEAN validation 3 s macro accuracy
  4. repeat for --seeds seeds, report mean and std (one run alone is noise: 0.53 vs 0.56 with the same code)

Outputs go to <KWS Capstone>/local_ml/runs/<time>/ (outside the repo): seed<k>.pt (weights + normalization), metrics.json

Run:
  python ML/Training/train.py
  python ML/Training/train.py --channels 24 --kernel 10x4 --specaug --mixup --seeds 5
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent                  # ML/Training
REPO = HERE.parents[1]
LOCAL = REPO.parent / "local_ml"                        # local only, never in the repo
DATASET = REPO / "ML" / "Dataset"
MANIFEST = DATASET / "manifest_1s.csv"
OUR_BIRDS = {"American Robin", "Baltimore Oriole", "Blue Jay", "Carolina Wren", "Eastern Phoebe", "Gray Catbird",
             "Red-tailed Hawk"}
sys.path.insert(0, str(REPO / "ML" / "Model"))
from model import DSCNN, activation_bytes               # noqa: E402

FEATURES = LOCAL / "features_june_1s.npz"
RUNS = LOCAL / "runs"

EPOCHS = 60
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
SHIFT_FRAMES = 10       # random time shift on train clips (frames are 10 ms)
MASKS = 2               # --specaug: up to this many time masks per clip
MASK_FRAMES = (2, 8)    # --specaug: each mask hides 2 to 8 frames
MIXUP_ALPHA = 0.4       # --mixup: Beta(alpha, alpha) mixing weight
# --wave augmentation (after PolyChirp, IEEE IS2 2026), applied to train audio only
WAVE_SHIFT = 0.10       # circular shift up to 10 % of the clip
WAVE_GAIN_DB = 10       # random gain +/- 10 dB
NOISE_P = 0.5           # chance a bird clip gets a random train no-bird clip mixed in
NOISE_SNR_DB = (5, 20)  # at this signal-to-noise ratio
WAVE_MASK_P = 0.25      # chance of zeroing one band of 2 to 8 % of the clip
WAVE_MASK = (0.02, 0.08)


def trusted_no_bird(rows):
    """True for every bird row, and for a no-bird row only when its 3 s chunk
      - comes from a recording where BirdNET found that recording's bird at least once (it gave >= 1 bird clip), and
      - is not right next to (+/- 3 s) a chunk where BirdNET heard one of our 7 birds (our-7 run, min conf 0.05).
    Otherwise BirdNET's "nothing here" is not reliable (audit: such clips are called a bird 2 to 3 x more often)."""
    bird_clips = defaultdict(int)
    for d in (DATASET / "Segments").iterdir():
        if d.name != "No bird":
            for f in d.glob("*.wav"):
                bird_clips[re.search(r"XC\d+", f.name).group(0)] += 1
    ours = defaultdict(set)
    for f in (DATASET / "Birdnet results our7 low").rglob("*.BirdNET.results.csv"):
        xc = re.match(r"XC\d+", f.name).group(0)
        with open(f, newline="", encoding="utf-8") as g:
            ours[xc] |= {round(float(r["Start (s)"]), 2) for r in csv.DictReader(g) if r["Common name"] in OUR_BIRDS}
    keep = []
    for r in rows:
        if r["label"] != "no_bird":
            keep.append(True)
            continue
        m = re.search(r"(XC\d+).*_(\d+\.\d+)s_(\d+\.\d+)s\.wav$", r["source_clip"])
        xc, start = m.group(1), round(float(m.group(2)), 2)
        keep.append(bird_clips[xc] > 0 and round(start - 3, 2) not in ours[xc] and round(start + 3, 2) not in ours[xc])
    return np.array(keep)


def load(path, device):
    """Features + labels on the device, normalized with train statistics, plus 3 s groups per split.
    val and test also get a "clean" subset (trusted labels only) for the official score."""
    d = np.load(path, allow_pickle=False)
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["path"] for r in rows] == [str(p) for p in d["path"]], "features and manifest_1s.csv rows differ"
    source = np.array([r["split"] + "/" + r["source_clip"] for r in rows])
    x = d["mfcc"].astype(np.float32)
    split = d["split"]
    trusted = trusted_no_bird(rows)
    mean = x[split == "train"].mean(axis=(0, 1))
    std = x[split == "train"].std(axis=(0, 1)) + 1e-6
    data = {}
    for s in ("train", "val", "test"):
        m = split == s                                                  # keeps manifest row order
        y = d["label"][m]
        _, group = np.unique(source[m], return_inverse=True)          # 3 s clip index per 1 s slice
        assert all(len(set(y[group == g])) == 1 for g in range(group.max() + 1)), "3 s group with two labels"
        data[s] = {
            "x": torch.from_numpy((x[m] - mean) / std).unsqueeze(1).to(device),   # (N, 1, frames, coeffs)
            "y": torch.from_numpy(y).long().to(device),
            "group": torch.from_numpy(group).long().to(device),
            "recording": d["recording"][m],
        }
        if s != "train":
            k = trusted[m]
            idx = torch.from_numpy(np.nonzero(k)[0]).to(device)
            data[s]["clean"] = {"idx": idx, "y": data[s]["y"][idx],
                                "group": torch.unique(data[s]["group"][idx], return_inverse=True)[1],
                                "recording": d["recording"][m][k]}
    names = [str(n) for n in d["label_names"]]
    data["norm"] = (torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device))
    data["names"] = names
    return data, names, mean, std


def add_waves(data, device):
    """--wave: load the train audio (16 kHz, 1 s) to the device in the same row order as the features."""
    import soundfile as sf
    sys.path.insert(0, str(LOCAL / "june_mfcc"))
    import mfcc_torch                                    # LOCAL port of June's MFCC (golden-checked vs MATLAB)
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    wav = np.stack([sf.read(REPO / "ML" / "Dataset" / r["path"], dtype="float32")[0] for r in rows])
    tr = data["train"]
    tr["wave"] = torch.from_numpy(wav).to(device)
    tr["noise_pool"] = torch.nonzero(tr["y"] == data["names"].index("no_bird")).squeeze(1)
    data["mfcc"] = (mfcc_torch.mfcc, mfcc_torch.build(device=device))


def rms(x):
    return x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)


def wave_batch(data, idx):
    """Augment the audio of train rows idx, then June's MFCC + train normalization -> (b, 1, 98, 13)."""
    tr = data["train"]
    x, y = tr["wave"][idx].clone(), tr["y"][idx]
    b, n = x.shape
    dev = x.device
    # background: a random train no-bird clip under bird clips at 5 to 20 dB SNR
    bird = (y != data["names"].index("no_bird")) & (torch.rand(b, device=dev) < NOISE_P)
    pool = tr["noise_pool"]
    noise = tr["wave"][pool[torch.randint(len(pool), (b,), device=dev)]]
    snr = torch.empty(b, 1, device=dev).uniform_(*NOISE_SNR_DB)
    noise = noise * rms(x) / rms(noise) / 10 ** (snr / 20)
    x = x + noise * bird[:, None]
    # circular shift
    shift = torch.randint(-int(WAVE_SHIFT * n), int(WAVE_SHIFT * n) + 1, (b, 1), device=dev)
    x = torch.gather(x, 1, (torch.arange(n, device=dev)[None] - shift) % n)
    # gain + polarity, then keep inside the ADC range
    gain = 10 ** (torch.empty(b, 1, device=dev).uniform_(-WAVE_GAIN_DB, WAVE_GAIN_DB) / 20)
    sign = torch.where(torch.rand(b, 1, device=dev) < 0.5, -1.0, 1.0)
    x = (x * gain * sign).clamp(-1.0, 1.0)
    # one zeroed band of 2 to 8 % of the clip
    w = (torch.empty(b, 1, device=dev).uniform_(*WAVE_MASK) * n).long()
    start = (torch.rand(b, 1, device=dev) * (n - w)).long()
    t = torch.arange(n, device=dev)[None]
    hit = (t >= start) & (t < start + w) & (torch.rand(b, 1, device=dev) < WAVE_MASK_P)
    x = x * ~hit
    fn, consts = data["mfcc"]
    mean, std = data["norm"]
    feats = fn(x, consts).float()
    return ((feats - mean) / std).unsqueeze(1)


def time_shift(x, max_shift):
    """Roll each clip in the batch along time by a random number of frames (one gather, no Python loop)."""
    if max_shift == 0:
        return x
    b, _, t, _ = x.shape
    shifts = torch.randint(-max_shift, max_shift + 1, (b, 1), device=x.device)
    idx = (torch.arange(t, device=x.device).unsqueeze(0) - shifts) % t           # (b, t)
    return torch.gather(x, 2, idx[:, None, :, None].expand_as(x))


def time_masks(x, n_masks, width):
    """SpecAugment in time: set up to n_masks random bands of frames to 0 (the normalized mean)."""
    b, _, t, _ = x.shape
    frames = torch.arange(t, device=x.device).unsqueeze(0)                       # (1, t)
    keep = torch.ones(b, t, dtype=torch.bool, device=x.device)
    for _ in range(n_masks):
        w = torch.randint(width[0], width[1] + 1, (b, 1), device=x.device)
        start = (torch.rand(b, 1, device=x.device) * (t - w + 1)).long()
        keep &= ~((frames >= start) & (frames < start + w))
    return x * keep[:, None, :, None]


def probs(model, x, batch=1024):
    model.eval()
    with torch.no_grad():
        return torch.cat([F.softmax(model(x[i:i + batch]), 1) for i in range(0, len(x), batch)])


def scores(p, y, group, n_cls):
    """1 s and 3 s confusion matrices. 3 s = mean softmax over the slices of one 3 s clip."""
    cm1 = confusion(y.cpu().numpy(), p.argmax(1).cpu().numpy(), n_cls)
    n_g = int(group.max()) + 1
    pg = torch.zeros(n_g, n_cls, device=p.device).index_add_(0, group, p)
    yg = torch.zeros(n_g, dtype=torch.long, device=p.device).scatter_(0, group, y)
    cm3 = confusion(yg.cpu().numpy(), pg.argmax(1).cpu().numpy(), n_cls)
    return cm1, cm3


def confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_acc(cm):
    return cm.diagonal() / np.maximum(cm.sum(1), 1)


def train_once(data, n_cls, cfg, seed, device, verbose=False):
    """One training run. cfg keys: channels, kernel, dropout, specaug, mixup, epochs, wave (needs add_waves).
    Returns results + best state."""
    torch.manual_seed(seed)
    model = DSCNN(n_classes=n_cls, channels=cfg["channels"], dropout=cfg["dropout"], first_kernel=cfg["kernel"],
                  first_stride=cfg.get("stride", (2, 2)), pool=cfg.get("pool", "avg"),
                  dilations=(1, 2, 4, 8) if cfg.get("dilated") else (1, 1, 1, 1)).to(device)
    counts = torch.bincount(data["train"]["y"], minlength=n_cls).float()
    w = (1.0 / counts) / (1.0 / counts).mean()
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    xtr, ytr = data["train"]["x"], data["train"]["y"]
    val = data["val"]
    best = {"val3c": -1.0}

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = torch.randperm(len(ytr), device=device)
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            yb = ytr[idx]
            xb = wave_batch(data, idx) if cfg.get("wave") else time_shift(xtr[idx], SHIFT_FRAMES)
            if cfg["specaug"]:
                xb = time_masks(xb, MASKS, MASK_FRAMES)
            opt.zero_grad()
            if cfg["mixup"]:
                lam = float(np.random.default_rng().beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm = torch.randperm(len(yb), device=device)
                out = model(lam * xb + (1 - lam) * xb[perm])
                loss = lam * loss_fn(out, yb) + (1 - lam) * loss_fn(out, yb[perm])
            else:
                loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        p = probs(model, val["x"])
        cm1, cm3 = scores(p, val["y"], val["group"], n_cls)
        c = val["clean"]
        _, cm3c = scores(p[c["idx"]], c["y"], c["group"], n_cls)
        val3, val3c = per_class_acc(cm3).mean(), per_class_acc(cm3c).mean()
        if val3c > best["val3c"]:
            best = {"val3c": float(val3c), "val3": float(val3), "val1": float(per_class_acc(cm1).mean()),
                    "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  seed {seed} epoch {epoch:3d}  val 1 s {per_class_acc(cm1).mean():.3f}  val 3 s {val3:.3f}  "
                  f"val 3 s clean {val3c:.3f}  (best clean {best['val3c']:.3f} @ {best['epoch']})")

    model.load_state_dict(best["state"])
    test = data["test"]
    p = probs(model, test["x"])
    cm1, cm3 = scores(p, test["y"], test["group"], n_cls)
    c = test["clean"]
    _, cm3c = scores(p[c["idx"]], c["y"], c["group"], n_cls)
    per_rec = defaultdict(list)
    for r, t, q in zip(c["recording"], c["y"].cpu().numpy(), p[c["idx"]].argmax(1).cpu().numpy()):
        per_rec[(str(r), int(t))].append(t == q)
    return {
        "seed": seed, "best_epoch": best["epoch"], "val1": best["val1"], "val3": best["val3"],
        "val3c": best["val3c"],
        "test1": float(per_class_acc(cm1).mean()), "test3": float(per_class_acc(cm3).mean()),
        "test3c": float(per_class_acc(cm3c).mean()),
        "test3_per_class": per_class_acc(cm3).tolist(), "test1_per_class": per_class_acc(cm1).tolist(),
        "test3c_per_class": per_class_acc(cm3c).tolist(),
        "test3_confusion": cm3.tolist(), "test3c_confusion": cm3c.tolist(),
        "worst_recordings": sorted((float(np.mean(v)), k[0], k[1], len(v)) for k, v in per_rec.items())[:5],
        "params": sum(q.numel() for q in model.parameters()),
        "ram_bytes": 2 * activation_bytes(model.cpu(), *data["train"]["x"].shape[2:]),
        "state": best["state"],
    }


def parse_kernel(s):
    a, b = s.lower().split("x")
    return (int(a), int(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--kernel", type=parse_kernel, default=(3, 3), help="first conv, frames x coeffs, e.g. 10x4")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--specaug", action="store_true")
    ap.add_argument("--mixup", action="store_true")
    ap.add_argument("--wave", action="store_true", help="augment train audio, MFCC on the GPU (local port)")
    ap.add_argument("--pool", choices=("avg", "attn"), default="avg")
    ap.add_argument("--dilated", action="store_true", help="depthwise blocks dilated 1, 2, 4, 8 in time")
    ap.add_argument("--stride", type=parse_kernel, default=(2, 2), help="first conv stride, frames x coeffs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    data, names, mean, std = load(args.features, device)
    n_cls = len(names)
    if args.wave:
        add_waves(data, device)
    cfg = {k: getattr(args, k) for k in ("channels", "kernel", "dropout", "specaug", "mixup", "epochs", "wave",
                                         "pool", "dilated", "stride")}
    splits = ("train", "val", "test")
    print({s: len(data[s]["y"]) for s in splits}, "1 s clips;",
          {s: int(data[s]["group"].max()) + 1 for s in splits}, "3 s clips;",
          {s: int(data[s]["clean"]["group"].max()) + 1 for s in ("val", "test")}, "clean 3 s clips;", "config", cfg)

    runs = [train_once(data, n_cls, cfg, seed, device, verbose=True) for seed in range(args.seeds)]
    for key, label in (("val3c", "val 3 s clean"), ("test3c", "TEST 3 s CLEAN (official)"),
                       ("test3", "test 3 s full"), ("test1", "test 1 s full")):
        v = [r[key] for r in runs]
        print(f"{label:26s} macro acc {np.mean(v):.3f} +/- {np.std(v):.3f}   ({', '.join(f'{x:.3f}' for x in v)})")
    pc = np.mean([r["test3c_per_class"] for r in runs], axis=0)
    print("test 3 s clean per class (mean over seeds):")
    for name, a in zip(names, pc):
        print(f"  {name:18s} {a:.3f}")
    print(f"params {runs[0]['params']:,}, two largest int8 feature maps {runs[0]['ram_bytes'] / 1024:.1f} KB")

    out = RUNS / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    for r in runs:
        torch.save({"state_dict": r["state"], "mean": mean, "std": std, "label_names": names,
                    "config": cfg, "features": str(args.features)}, out / f"seed{r['seed']}.pt")
    (out / "metrics.json").write_text(json.dumps(
        {"config": cfg, "label_names": names, "runs": [{k: v for k, v in r.items() if k != "state"} for r in runs]},
        indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
