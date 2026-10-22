"""
Turn the 3 s BirdNET clips in Segments/ into 1 s training examples in clips_1s/.

Steps for every 3 s clip:
  1. mono + resample to 16 kHz
  2. cut into 1 s slices (0-1, 1-2, 2-3 s, no overlap)
  3. bird slices: keep only if they hold a call (1-8 kHz peak >= 10 dB over the clip's own quiet level)
     no-bird slices: always keep
  4. loudness: scale every slice to the same RMS level
  5. write the slice and one manifest row (label, split, source recording)

Before cutting, each recording is capped at MAX_CLIPS_PER_REC 3 s clips per class, and every recording
is put in exactly one split (train / val / test), so no recording is seen in two splits.

Run from the repo root or anywhere:
  python ML/Dataset/make_clips_1s.py            # writes clips_1s/ and manifest_1s.csv
  python ML/Dataset/make_clips_1s.py --dry-run  # only prints the counts, writes nothing
"""
import argparse
import csv
import random
import re
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent          # ML/Dataset
SEGMENTS = HERE / "Segments"                    # input: 3 s clips, one folder per class
RAW = HERE / "raw"                              # used only to know which bird each recording belongs to
OUT = HERE / "clips_1s"                         # output: clips_1s/<split>/<label>/*.wav
MANIFEST = HERE / "manifest_1s.csv"

SR = 16000                  # target sample rate (Nyquist 8 kHz)
SLICE_S = 1.0               # slice length
MAX_CLIPS_PER_REC = 25      # cap per recording per class (3 s clips)
BAND = (1000, 8000)         # where our birds' calls live
CALL_MARGIN_DB = 10.0       # slice peak must be this far above the clip's quiet level
FLOOR_PERCENTILE = 10       # "quiet level" = 10th percentile of frame energy in the clip
TARGET_RMS_DB = -20.0       # every slice is scaled to this RMS (dBFS)
VAL_SHARE, TEST_SHARE = 0.15, 0.15
MIN_VAL_RECS, MIN_TEST_RECS = 2, 2
SEED = 0

# Segments folder name (BirdNET's common name) -> label used for training
LABELS = {
    "American Robin": "american_robin",
    "Baltimore Oriole": "baltimore_oriole",
    "Blue Jay": "blue_jay",
    "Carolina Wren": "carolina_wren",
    "Eastern Phoebe": "eastern_phoebe",
    "Gray Catbird": "gray_catbird",
    "Red-tailed Hawk": "red_tailed_hawk",
    "No bird": "no_bird",
}
# raw/ folder name (team naming) -> label
RAW_TO_LABEL = {
    "American_Robin": "american_robin",
    "Baltimore_Oriole": "baltimore_oriole",
    "Blue_Jay": "blue_jay",
    "Carolina_Wren": "carolina_wren",
    "Eastern_Phoebe": "eastern_phoebe",
    "Grey_Catbird": "gray_catbird",
    "Red-tailed_Hawk": "red_tailed_hawk",
}


def xc_id(name):
    m = re.search(r"XC(\d+)", name)
    return "XC" + m.group(1)


def birdnet_conf(name):
    """BirdNET clip names start with the confidence, e.g. 0.912_3_XC... ; no-bird clips have none."""
    try:
        return float(name.split("_")[0])
    except ValueError:
        return None


def load_16k_mono(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)   # has its own low-pass, so no aliasing
    return x.astype(np.float32)


def band_energy_db(x, n_fft=512, hop=256):
    """Energy in BAND for each 32 ms frame (16 ms hop), in dB. Same framing as our MFCC plan."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.stack([x[i * hop:i * hop + n_fft] for i in range(n_frames)])
    spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    return 10 * np.log10(spec[:, band].sum(axis=1) + 1e-12), hop


def normalize(x):
    rms = np.sqrt(np.mean(x ** 2))
    if rms < 1e-7:                       # digital silence: nothing to scale
        return None
    y = x * (10 ** (TARGET_RMS_DB / 20) / rms)
    peak = np.max(np.abs(y))
    if peak > 0.99:                      # never clip: back off so the peak fits
        y *= 0.99 / peak
    return y


def recording_owner():
    """XC id -> the bird folder in raw/ it was downloaded for (that bird decides its split)."""
    owner = {}
    for d in RAW.iterdir():
        for f in d.iterdir():
            owner[xc_id(f.name)] = d.name
    return owner


def assign_splits(owner, weight):
    """Put every recording in one split so each bird gets ~15 % of its clips in val and ~15 % in test.

    Recordings give very different numbers of clips, so picking recordings at random makes lopsided
    splits. Instead, per bird: shuffle its recordings, then hand each one to the split that is furthest
    below its share of clips. Afterwards make sure val and test hold at least 2 recordings each.
    weight[rec] = how many clips the recording gives (its own bird's clips, after the cap)."""
    shares = {"train": 1 - VAL_SHARE - TEST_SHARE, "val": VAL_SHARE, "test": TEST_SHARE}
    by_bird = defaultdict(list)
    for rec, bird in owner.items():
        by_bird[bird].append(rec)
    rng = random.Random(SEED)
    split = {}
    for bird in sorted(by_bird):
        recs = sorted(by_bird[bird])
        rng.shuffle(recs)
        recs.sort(key=lambda r: -weight.get(r, 0))      # big recordings first, ties stay shuffled
        total = sum(weight.get(r, 0) for r in recs) or 1
        got = {s: 0 for s in shares}
        members = {s: [] for s in shares}
        for rec in recs:
            s = max(shares, key=lambda s: shares[s] * total - got[s])
            got[s] += weight.get(rec, 0)
            members[s].append(rec)
        for s, need in (("test", MIN_TEST_RECS), ("val", MIN_VAL_RECS)):
            while len(members[s]) < need:                 # borrow the smallest train recording
                rec = min(members["train"], key=lambda r: weight.get(r, 0))
                members["train"].remove(rec)
                members[s].append(rec)
        for s, rs in members.items():
            for rec in rs:
                split[rec] = s
    return split


def pick_clips(files):
    """Cap each recording at MAX_CLIPS_PER_REC: keep the highest BirdNET scores (random for no-bird)."""
    by_rec = defaultdict(list)
    for f in files:
        by_rec[xc_id(f.name)].append(f)
    rng = random.Random(SEED)
    kept = []
    for rec in sorted(by_rec):
        fs = sorted(by_rec[rec])
        if birdnet_conf(fs[0].name) is None:
            rng.shuffle(fs)
        else:
            fs.sort(key=lambda f: -birdnet_conf(f.name))
        kept += fs[:MAX_CLIPS_PER_REC]
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print counts only, write nothing")
    args = ap.parse_args()
    if not args.dry_run and OUT.exists():
        raise SystemExit(f"{OUT} already exists. Delete it first if you want to rebuild.")

    owner = recording_owner()
    n_len = int(SR * SLICE_S)
    rows, stats = [], defaultdict(lambda: defaultdict(int))

    # cap first, then count clips per recording (own bird only) to balance the splits
    picked, weight = {}, defaultdict(int)
    for folder, label in LABELS.items():
        files = sorted((SEGMENTS / folder).glob("*.wav"))
        picked[label] = pick_clips(files)
        stats[label]["clips_3s"] = len(files)
        stats[label]["clips_after_cap"] = len(picked[label])
        for f in picked[label]:
            rec = xc_id(f.name)
            if RAW_TO_LABEL[owner[rec]] == label:
                weight[rec] += 1
    split = assign_splits(owner, weight)

    for folder, label in LABELS.items():
        for f in picked[label]:
            rec = xc_id(f.name)
            sp = split[rec]
            x = load_16k_mono(f)
            n_slices = len(x) // n_len                       # short clips give only their full seconds
            if n_slices == 0:
                continue
            e_db, hop = band_energy_db(x)
            floor = np.percentile(e_db, FLOOR_PERCENTILE)
            slices = []
            for k in range(n_slices):
                a, b = k * n_len, (k + 1) * n_len
                fr = e_db[a // hop:max(a // hop + 1, (b - 512) // hop + 1)]
                slices.append((k, x[a:b], float(fr.max() - floor)))
            if label == "no_bird":
                keep = slices
            else:
                keep = [s for s in slices if s[2] >= CALL_MARGIN_DB]
                if not keep:                                  # BirdNET heard a bird: keep the loudest slice
                    keep = [max(slices, key=lambda s: s[2])]
            for k, seg, margin in keep:
                y = normalize(seg)
                if y is None:
                    stats[label]["dropped_silent"] += 1
                    continue
                name = f"{f.stem}_s{k}.wav"
                path = OUT / sp / label / name
                if not args.dry_run:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path, y, SR, subtype="PCM_16")
                stats[label][sp] += 1
                rows.append({"path": path.relative_to(HERE).as_posix(), "label": label, "split": sp,
                             "recording": rec, "recording_bird": owner[rec], "source_clip": f.name,
                             "slice": k, "birdnet_conf": birdnet_conf(f.name), "margin_db": round(margin, 1)})
            stats[label]["dropped_no_call"] += len(slices) - len(keep)
        print(f"done {label:17s} ({len(rows)} slices so far)", flush=True)

    if not args.dry_run:
        with open(MANIFEST, "w", newline="", encoding="utf-8") as g:
            w = csv.DictWriter(g, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"{'label':17s} {'3s':>5s} {'capped':>7s} {'train':>6s} {'val':>5s} {'test':>5s} {'no-call':>8s}")
    for label in LABELS.values():
        s = stats[label]
        print(f"{label:17s} {s['clips_3s']:5d} {s['clips_after_cap']:7d} {s['train']:6d} {s['val']:5d} "
              f"{s['test']:5d} {s['dropped_no_call']:8d}")
    print(f"total slices: {len(rows)}" + ("  (dry run, nothing written)" if args.dry_run else f"  -> {OUT}"))


if __name__ == "__main__":
    main()
