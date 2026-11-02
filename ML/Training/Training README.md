###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# Training (`train.py`, `sweep.py`)

## Input
- `ML/Dataset/manifest_1s.csv` (labels, splits, source clips) and the BirdNET results folders (for the trusted
  no-bird rule).
- A features file with June's MFCC for every row of the manifest, same order: `mfcc (N, 98, 13)`, `label`,
  `label_names`, `split`, `recording`, `path`. It is made from June's code in `Firmware/Features/` and kept
  **outside the repo** (default `<KWS Capstone>/local_ml/features_june_1s.npz`; pass another with `--features`).
- PyTorch, with CUDA if available (the whole dataset is put on the GPU, so training is fast).

## Run the current best model
```
python ML/Training/train.py --channels 24 --specaug --mixup --pool attn --dilated
```
Note: the default `--channels` is 32, so pass `--channels 24` for the chip-sized model.

Other flags: `--seeds N` (default 3), `--epochs` (60), `--kernel 10x4`, `--stride 2x4`, `--dropout`,
`--wave` (augment the audio and compute MFCC on the GPU; needs a local PyTorch copy of June's MFCC, not in the repo).

## What it does
1. Normalizes each of the 13 coefficients with the **train** mean and std (the chip must use the same numbers; they
   are saved with the weights).
2. Cross-entropy with class weights 1/count; Adam, learning rate 0.001, cosine decay, weight decay 0.0001, batch 64.
3. Augmentation on train only: time shift up to 10 frames, `--specaug` (up to 2 masks of 2 to 8 frames), `--mixup`.
4. Keeps the epoch with the best **clean validation** 3 s score; repeats for each seed and prints mean +/- std.

## How it is scored
- **Macro accuracy**: mean of the 8 per-class accuracies.
- **3 s decision**: softmax averaged over the 1 s slices of one 3 s source clip (the device's decision).
- **Clean (official)**: no-bird val/test clips are scored only when their label can be trusted
  (`trusted_no_bird()`: the recording gave at least one bird clip, and the clip is not within 3 s of a chunk where
  BirdNET heard one of our 7 at 0.05). A label audit of test errors showed many untrusted no-bird clips contain our
  birds. The rule only changes which clips are scored; training uses every clip. The score on all clips is printed too.

## Outputs
`<KWS Capstone>/local_ml/runs/<time>/` (outside the repo): `seed<k>.pt` (weights, normalization, config) and
`metrics.json` (per seed: val/test scores, per class, confusion matrices, worst test recordings).

## Current results (best model, 3 seeds)
| Score | Value |
|---|---|
| Test, 3 s, clean (official) | 0.805 +/- 0.015 (std over 3 seeds) |
| Validation, 3 s, clean | 0.814 |
| Test, 3 s, all clips | 0.789 |
| Test, 1 s, all clips | 0.655 |

Per class (test, 3 s, clean): Robin 0.77, Oriole 0.82, Blue Jay 0.88, Wren 0.75, Phoebe 0.89, Catbird 0.72,
Hawk 0.82, No bird 0.79. Team target: 0.90.

## Sweeps
```
python ML/Training/sweep.py
```
Runs a grid of settings (channels, first kernel, dropout, SpecAugment, mixup), 3 seeds each, and writes one CSV
row per setting to `<KWS Capstone>/local_ml/runs/sweep_<time>.csv`. Edit `GRID` in the file to change what is tried.
