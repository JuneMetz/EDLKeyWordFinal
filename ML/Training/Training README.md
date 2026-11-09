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

## Run the current best model (BirdTCN)
```
python ML/Training/train.py --specaug --mixup
```
The defaults are the best model: `--arch tcn`, `--channels 64`, `--tcn-kernel 9`, each clip's mean removed.

Old best (2D DS-CNN, test 0.805):
```
python ML/Training/train.py --arch ds2d --channels 24 --pool attn --dilated --specaug --mixup --no-cmn
```

Other flags: `--seeds N` (default 3), `--epochs` (60), `--dropout`, `--no-cmn` (keep each clip's mean),
`--wave` (augment the audio and compute MFCC on the GPU; needs a local PyTorch copy of June's MFCC, not in the repo).
DS-CNN only: `--kernel 10x4`, `--stride 2x4`, `--pool`, `--dilated`.

## What it does
1. Normalizes each of the 13 coefficients with the **train** mean and std (saved with the weights), then subtracts
   each clip's own mean per coefficient over its 98 frames (removes the steady colouring of the microphone and site;
   the chip must do the same).
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

## Current results (3 seeds, +/- = std over seeds)
| Score | BirdTCN (current) | Old DS-CNN |
|---|---|---|
| Test, 3 s, clean (official) | **0.854 +/- 0.010** | 0.805 +/- 0.015 |
| Validation, 3 s, clean | 0.855 | 0.814 |
| Test, 3 s, all clips | 0.840 | 0.789 |
| Test, 1 s, all clips | 0.699 | 0.655 |
| Weights / peak int8 RAM | 27,913 / 6.1 KB | 4,041 / 16.1 KB |

Per class, BirdTCN (test, 3 s, clean): Robin 0.89, Oriole 0.83, Blue Jay 0.89, Wren 0.84, Phoebe 0.93,
Catbird 0.85, Hawk 0.81, No bird 0.80. Team target: 0.90.

How BirdTCN was chosen (clean validation only, 3 seeds each; test read once at the end): old DS-CNN 0.807 ->
time-only 64 ch 0.845 -> + clip mean removed 0.857 -> + kernel 9 0.862. No gain from 96, 128 or 256 channels
(0.853 to 0.857) or from 40 log-mel bands instead of MFCC (0.842). See `ML/Figures/10_validation_experiments.png`.
A simulated int8 version (8-bit weights and feature maps) scored 0.859 vs 0.860 in float on the clean test.
Note: one seed's clean test score moves about +/-0.04 if the test recordings are resampled, so small differences
between settings are noise; compare settings on the same clips.

## Sweeps
```
python ML/Training/sweep.py
```
Runs a grid of settings for the old DS-CNN (channels, first kernel, dropout, SpecAugment, mixup), 3 seeds each, and writes one CSV
row per setting to `<KWS Capstone>/local_ml/runs/sweep_<time>.csv`. Edit `GRID` in the file to change what is tried.
