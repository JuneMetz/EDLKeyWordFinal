###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# Training (`train.py`)

## Input
- `ML/Dataset/manifest_1s.csv` (labels, splits, source clips) and the BirdNET results folders (for the trusted
  no-bird rule).
- **Features file**, kept **outside the repo** because of its size (157 MB): one row per row of the manifest, same
  order, arrays `mfcc (N, 98, 13)`, `label`, `label_names`, `split`, `recording`, `path`. Default
  `<KWS Capstone>/local_ml/features_pcen_fastT15_1s.npz`; pass another with `--features`.
- **Extra training clips** from iNaturalist (Oriole and Hawk, plus m4a Oriole recordings), also outside the repo, in
  `<KWS Capstone>/local_ml/extra_train/` (`mfcc` + `label` only). They are used for training only, never for
  validation or test. Which recordings and 1 s slices they are: `ML/Dataset/inaturalist_recordings.csv`,
  `ML/Dataset/manifest_inat_1s.csv`.
- PyTorch, with CUDA if available (the whole dataset is put on the GPU, so training is fast).

## Features: June's MFCC with PCEN
The features follow June's MFCC recipe (`Firmware/Features/`: 30 ms frames, 10 ms hop, 40 mel bands, DCT, 13
coefficients) with one change: her **log step is replaced by PCEN** (per-channel energy normalization). For each mel
band, PCEN divides the energy by a slowly moving average of that band's own recent energy, then compresses it:
```
M[t] = (1 - s) M[t-1] + s E[t]                       (smoothed background, time constant T)
P[t] = (E[t] / (eps + M[t])^alpha + delta)^r - delta^r
```
Settings: **alpha 0.8, delta 10, r 0.25, T 0.15 s** (eps 1e-6; s from T and the 10 ms hop, librosa's formula). A
steady background (wind, hum, a site's noise) is divided away, while a call that rises above it stands out. On
validation, PCEN beat the log step (0.867 vs 0.855) and on test 0.873 vs 0.854.
The feature code is a local PyTorch port of June's MFCC and stays out of the repo (June owns the recipe). On the chip,
PCEN must be added to June's C version.

## Make the final model (3 steps)
Run from the repo root. Each run trains 3 seeds and saves to `<KWS Capstone>/local_ml/runs/<time>_<name>/`.

1. **Teacher** (large model, laptop only; about 7 minutes per seed on an RTX 5070, run alone):
   ```
   python ML/Training/train.py --model teacher --batch 64 --extra "<KWS Capstone>/local_ml/extra_train/features_pcen_fastT15_inat.npz" --name teacher
   ```
2. **Check the recipe on validation** (best epoch picked by clean validation, 60 epochs):
   ```
   python ML/Training/train.py --teacher <teacher run folder> --name student_val
   ```
   This gave validation 0.894 +/- 0.005, best epochs 22, 40 and 52.
3. **Final weights** (train + val + extras, fixed 38 epochs = the mean best epoch of step 2, last epoch kept, one
   test read):
   ```
   python ML/Training/train.py --teacher <teacher run folder> --final --name final
   ```
   Copy the 3 `seed*.pt` and `metrics.json` to `ML/Model/weights/`.

Flags: `--features`, `--extra` (train-only features files; `path:label_name` keeps one label), `--model student|teacher`,
`--teacher <run folder>`, `--final`, `--epochs`, `--batch` (default 32), `--seeds` (default 3), `--name`.

## What it does
1. Normalizes each of the 13 coefficients with the **train** mean and std (saved with the weights), then subtracts
   each clip's own mean per coefficient over its 98 frames (removes the steady colouring of the microphone and site;
   the chip must do the same).
2. Cross-entropy with class weights 1/count; Adam, learning rate 0.001, cosine decay, weight decay 0.0001, batch 32.
3. Augmentation on train only: time shift up to 10 frames, 2 time masks of 2 to 8 frames, mixup (alpha 0.4).
4. **Distillation** (with `--teacher`): the loss is 0.1 x the normal loss + 0.9 x how far BirdTCN's answers are from
   the teacher's (KL divergence), both softened with temperature 2. The teacher is the average of its 3 seeds and sees
   the same augmented batch. This teaches BirdTCN how sure to be and which birds sound alike, not only the right label.
5. Without `--final`: keeps the epoch with the best clean validation score. With `--final`: keeps the last epoch.

## How it is scored
- **Macro accuracy**: mean of the 8 per-class accuracies.
- **3 s decision**: softmax averaged over the 1 s slices of one 3 s source clip (the device's decision).
- **Clean (official)**: no-bird val/test clips are scored only when their label can be trusted
  (`trusted_no_bird()`: the recording gave at least one bird clip, and the clip is not within 3 s of a chunk where
  BirdNET heard one of our 7 at 0.05). A label audit of test errors showed many untrusted no-bird clips contain our
  birds. The rule only changes which clips are scored; training uses every clip. The score on all clips is printed too.

## Results (3 seeds, +/- = std over seeds)
| Score | Final model | BirdTCN, log MFCC (11-09) | First DS-CNN (11-01) |
|---|---|---|---|
| Test, 3 s, clean (official) | **0.901 +/- 0.006** | 0.854 +/- 0.010 | 0.805 +/- 0.015 |
| Test, 3 s, all clips | 0.883 | 0.840 | 0.789 |
| Test, 1 s, all clips | 0.743 | 0.699 | 0.655 |
| Simulated 8-bit, 3 s clean | 0.901 to 0.903 | 0.859 (earlier run of the same recipe, 0.860 float) | |
| Weights / peak 8-bit RAM | 27,913 / 6.1 KB | 27,913 / 6.1 KB | 4,041 / 16.1 KB |

Per class, final model (test, 3 s, clean): Robin 0.93, Oriole 0.80, Blue Jay 0.94, Wren 0.93, Phoebe 0.99,
Catbird 0.94, Hawk 0.89, No bird 0.80. Team target: 0.90.

What each step added (paired on the same test clips where possible): BirdTCN with clip mean removed 0.805 -> 0.854;
PCEN 0.873; iNaturalist Oriole and Hawk 0.880; training on train + val 0.895; m4a Oriole recordings + batch 32 +
distillation 0.901. Most other ideas (noise mixing, frame sizes, wider models, other training settings, other decision
rules, no-bird fixes) stayed within noise: see `ML/Figures/10_validation_experiments.png` and
`ML/Figures/13_all_models.png`.

Be honest about these numbers: the test set was read many times during the project and the clean rule came from a
study of test errors, so 0.901 is somewhat optimistic for new recordings. One seed's score moves about +/- 0.04 if the
test recordings are resampled, so differences under about 0.01 are noise. The weakest classes are no bird (mostly
label noise: about half its errors name the recording's own bird, see `ML/Figures/12_nobird_errors.png`) and Oriole
(only 20 test recordings).
