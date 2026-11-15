"""
Train the bird classifier (BirdTCN, ML/Model/model.py) on precomputed features. Final model: test 0.901.

Input: a features .npz made OUTSIDE this repo (June's MFCC recipe with PCEN in place of the log step), one row per
1 s clip of ML/Dataset/manifest_1s.csv (same row order), with arrays:
  mfcc (N, 98, 13), label (N,), label_names (8,), split (N,) train/val/test, recording (N,), path (N,)
Default path: <KWS Capstone>/local_ml/features_pcen_fastT15_1s.npz (next to this repo, not in it).
Extra TRAINING data (iNaturalist recordings, listed in ML/Dataset/manifest_inat.csv and manifest_m4a.csv) comes as
features files with mfcc + label only; they are never used for validation or test.

Three steps make the final model (commands in ML/Training/Training README.md):
  1. --model teacher           train the large teacher (laptop only) on train + iNaturalist, best epoch by validation
  2. --teacher <dir>           check the student recipe on validation: BirdTCN distilled from the teacher, 60 epochs,
                               best epoch by clean validation
  3. --teacher <dir> --final   the final weights: train + val + extras, a fixed 38 epochs (the mean best epoch of
                               step 2), last epoch kept, then ONE test read

Scores reported:
  1 s macro acc  every 1 s clip on its own
  3 s macro acc  the device decision: softmax averaged over the 1 s slices cut from the same 3 s BirdNET clip
                 (manifest column source_clip)
  clean          the OFFICIAL score: same, but no-bird val/test clips are kept only when their label can be trusted
                 (trusted_no_bird below). Training still uses every clip. "all" = every val/test clip.
"macro" = mean of the 8 per-class accuracies, so the big no-bird class cannot hide a weak bird.

Training recipe:
  - normalize every coefficient with the TRAIN mean and std (saved with the weights; the chip must use the same
    numbers), then subtract each clip's own mean per coefficient over its 98 frames
  - cross-entropy with class weights 1/count, Adam lr 0.001, weight decay 0.0001, cosine decay, batch 32
  - augmentation on train only: time shift up to 10 frames, 2 time masks of 2 to 8 frames, mixup (alpha 0.4)
  - distillation (when --teacher is given): loss = 0.1 x cross-entropy + 0.9 x T^2 x KL(teacher || student) at
    temperature T = 2; the teacher is the mean of its 3 seeds and sees the same augmented batch
  - 3 seeds, mean and std reported

Outputs go to <KWS Capstone>/local_ml/runs/<time>_<name>/ (outside the repo): seed<k>.pt + metrics.json
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
from model import BirdTCN, TeacherTCN, chip_cost        # noqa: E402

FEATURES = LOCAL / "features_pcen_fastT15_1s.npz"
EXTRA = [f"{LOCAL / 'extra_train' / 'features_pcen_fastT15_inat.npz'}",                       # iNaturalist wav/mp3
         f"{LOCAL / 'extra_train' / 'features_pcen_fastT15_m4a.npz'}:baltimore_oriole"]        # iNaturalist m4a, Oriole only
RUNS = LOCAL / "runs"

EPOCHS = 60             # validation runs (best epoch picked by clean validation)
FINAL_EPOCHS = 38       # --final: mean best epoch of the validation runs (22, 40, 52)
BATCH = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SHIFT_FRAMES = 10       # random time shift on train clips (frames are 10 ms)
MASKS = 2               # up to this many time masks per clip
MASK_FRAMES = (2, 8)    # each mask hides 2 to 8 frames
MIXUP_ALPHA = 0.4       # Beta(alpha, alpha) mixing weight
KD_ALPHA = 0.9          # share of the loss that comes from the teacher
KD_TEMP = 2.0           # softening temperature for teacher and student


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


def remove_clip_mean(x):
    """From every coefficient subtract its own mean over the frames of that clip (x: ..., frames, coeffs).
    Removes the steady colouring of the microphone and the site; the chip must do the same."""
    return x - x.mean(axis=-2, keepdims=True)


def load(path, device):
    """Features + labels on the device, normalized with train statistics and clip mean removed, plus 3 s groups per
    split. val and test also get a "clean" subset (trusted labels only) for the official score."""
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
        xs = remove_clip_mean((x[m] - mean) / std)
        _, group = np.unique(source[m], return_inverse=True)          # 3 s clip index per 1 s slice
        assert all(len(set(y[group == g])) == 1 for g in range(group.max() + 1)), "3 s group with two labels"
        data[s] = {
            "x": torch.from_numpy(xs).unsqueeze(1).to(device),                    # (N, 1, frames, coeffs)
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
    return data, names, mean, std


def split_extra(spec):
    """"path" -> (path, ""); "path:label_name" -> (path, label_name). A drive letter like C: is not a label."""
    head, sep, tail = spec.rpartition(":")
    if sep and tail and not any(ch in tail for ch in "\\/."):
        return head, tail
    return spec, ""


def load_extra(spec, names, mean, std, device):
    """Extra training clips: "path" or "path:label_name" (keep only that label). Same normalization as load()."""
    path, only = split_extra(spec)
    d = np.load(path)
    keep = np.ones(len(d["label"]), bool) if not only else d["label"] == names.index(only)
    x = remove_clip_mean((d["mfcc"][keep].astype(np.float32) - mean) / std)
    return (torch.from_numpy(x).unsqueeze(1).to(device), torch.from_numpy(d["label"][keep].astype(np.int64)).to(device))


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


def clean_val_score(model, val, n_cls):
    c = val["clean"]
    _, cm3c = scores(probs(model, val["x"][c["idx"]]), c["y"], c["group"], n_cls)
    return float(per_class_acc(cm3c).mean())


def load_teachers(run_dir, device):
    """The teacher run's seed*.pt files, in eval mode. Their softmax is averaged."""
    teachers = []
    for f in sorted(Path(run_dir).glob("seed*.pt")):
        t = TeacherTCN().to(device)
        t.load_state_dict(torch.load(f, weights_only=False)["state_dict"])
        teachers.append(t.eval())
    assert teachers, f"no seed*.pt in {run_dir}"
    return teachers


def train_once(kind, x, y, n_cls, seed, epochs, batch, device, teachers=None, val=None):
    """One seed. kind "student" (BirdTCN) or "teacher" (TeacherTCN). With val: keep the epoch with the best clean
    validation 3 s score; without val (--final): keep the last epoch. Returns (model, best epoch)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = (BirdTCN(n_classes=n_cls) if kind == "student" else TeacherTCN(n_classes=n_cls)).to(device)
    counts = torch.bincount(y, minlength=n_cls).float()
    ce = nn.CrossEntropyLoss(weight=(1.0 / counts) / (1.0 / counts).mean())
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = (-1.0, None, epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y), device=device)
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            yb = y[b]
            xb = time_masks(time_shift(x[b], SHIFT_FRAMES), MASKS, MASK_FRAMES)
            lam = float(rng.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            perm = torch.randperm(len(yb), device=device)
            xmix = lam * xb + (1 - lam) * xb[perm]
            out = model(xmix)
            loss = lam * ce(out, yb) + (1 - lam) * ce(out, yb[perm])
            if teachers:
                with torch.no_grad():
                    pt = torch.stack([F.softmax(t(xmix) / KD_TEMP, 1) for t in teachers]).mean(0)
                kd = F.kl_div(F.log_softmax(out / KD_TEMP, 1), pt, reduction="batchmean") * KD_TEMP * KD_TEMP
                loss = (1 - KD_ALPHA) * loss + KD_ALPHA * kd
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if val is not None:
            v = clean_val_score(model, val, n_cls)
            if v > best[0]:
                best = (v, {k: t.detach().clone() for k, t in model.state_dict().items()}, epoch)
            if epoch % 10 == 0:
                print(f"  seed {seed} epoch {epoch:3d}  val 3 s clean {v:.3f}  (best {best[0]:.3f} @ {best[2]})", flush=True)
    if val is not None:
        model.load_state_dict(best[1])
    return model.eval(), best[2]


def test_scores(model, data, n_cls):
    te = data["test"]
    p = probs(model, te["x"])
    cm1, cm3 = scores(p, te["y"], te["group"], n_cls)
    c = te["clean"]
    _, cm3c = scores(p[c["idx"]], c["y"], c["group"], n_cls)
    return {"test3c": float(per_class_acc(cm3c).mean()), "test3": float(per_class_acc(cm3).mean()),
            "test1": float(per_class_acc(cm1).mean()), "test3c_per_class": per_class_acc(cm3c).tolist(),
            "test3c_confusion": cm3c.tolist(), "test3_confusion": cm3.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--extra", nargs="*", default=EXTRA, help='extra train-only features: "path" or "path:label_name"')
    ap.add_argument("--model", choices=("student", "teacher"), default="student")
    ap.add_argument("--teacher", type=Path, help="teacher run folder (seed*.pt): distill from it")
    ap.add_argument("--final", action="store_true", help="train on train + val, fixed epochs, keep the last epoch")
    ap.add_argument("--epochs", type=int, help=f"default {EPOCHS}, or {FINAL_EPOCHS} with --final")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--name", default="", help="suffix for the run folder")
    args = ap.parse_args()
    epochs = args.epochs or (FINAL_EPOCHS if args.final else EPOCHS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    data, names, mean, std = load(args.features, device)
    n_cls = len(names)
    parts = [(data["train"]["x"], data["train"]["y"])]
    if args.final:
        parts.append((data["val"]["x"], data["val"]["y"]))
    parts += [load_extra(e, names, mean, std, device) for e in args.extra]
    x = torch.cat([p[0] for p in parts])
    y = torch.cat([p[1] for p in parts])
    teachers = load_teachers(args.teacher, device) if args.teacher else None
    cfg = {"model": args.model, "final": args.final, "epochs": epochs, "batch": args.batch,
           "extra": [Path(split_extra(e)[0]).name + (":" + split_extra(e)[1] if split_extra(e)[1] else "") for e in args.extra],
           "features": args.features.name, "distilled": bool(teachers), "kd_alpha": KD_ALPHA if teachers else 0.0,
           "kd_temp": KD_TEMP}
    print(f"training clips: {len(y):,} ({'train + val' if args.final else 'train'} + extras); config {cfg}", flush=True)

    out = RUNS / (time.strftime("%Y%m%d_%H%M%S") + ("_" + args.name if args.name else ""))
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in range(args.seeds):
        model, ep = train_once(args.model, x, y, n_cls, seed, epochs, args.batch, device, teachers,
                               val=None if args.final else data["val"])
        r = {"seed": seed, "epoch": ep, **test_scores(model, data, n_cls)}
        if not args.final:
            r["val3c"] = clean_val_score(model, data["val"], n_cls)
        runs.append(r)
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "mean": mean, "std": std,
                    "label_names": names, "config": cfg}, out / f"seed{seed}.pt")
        print(f"seed {seed}: " + (f"val 3 s clean {r['val3c']:.3f} @ epoch {ep} | " if not args.final else "")
              + f"TEST 3 s clean {r['test3c']:.3f} | 3 s all {r['test3']:.3f} | 1 s {r['test1']:.3f}", flush=True)

    f = lambda k: f"{np.mean([r[k] for r in runs]):.3f} +/- {np.std([r[k] for r in runs]):.3f}"
    if not args.final:
        print(f"val 3 s clean (choose settings by this) {f('val3c')}, best epochs {[r['epoch'] for r in runs]}")
    print(f"TEST 3 s clean (official) {f('test3c')} | 3 s all {f('test3')} | 1 s {f('test1')}")
    print("per class:", " ".join(f"{n[:7]} {a:.2f}" for n, a in zip(names, np.mean([r["test3c_per_class"] for r in runs], 0))))
    if args.model == "student":
        w, macs, ram = chip_cost(BirdTCN(n_classes=n_cls))
        print(f"chip: {w:,} weights, {macs / 1e6:.2f} M multiply-adds per 1 s clip, peak RAM {ram / 1024:.1f} KB int8")
    (out / "metrics.json").write_text(json.dumps({"config": cfg, "label_names": names, "runs": runs}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
