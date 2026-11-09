###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# ML: bird classifier

Every second of audio becomes June's MFCC grid (98 x 13); each clip's own mean is removed; a small time-only CNN
(BirdTCN, 27,913 weights, 6.1 KB peak RAM) gives 8 scores (7 birds + no bird); the device averages them over 3 s
and logs the top class.

| Folder | What is there |
|---|---|
| `Dataset/` | how the labelled 1 s clips were built (xeno-canto + BirdNET), `make_clips_1s.py`, manifest, BirdNET results |
| `Model/` | `model.py`: BirdTCN (current) and the old DS-CNN |
| `Training/` | `train.py` (training + scoring), `sweep.py` (grid of settings for the old DS-CNN) |
| `Inference/` | plan for running the model on the nRF52805 (not started) |
| `Figures/` | diagrams and result plots (pipeline, architecture, results) |

**Current result:** 0.854 macro accuracy on the 3 s decision (official clean test, 3 seeds), up from 0.805 with the
old DS-CNN; target 0.90. The weakest classes are no bird, Hawk and Oriole. More recordings is the next lever:
a bigger model did not help.

Data on disk (`Dataset/`, not in git): `raw/` = the 1,280 downloaded recordings, `Segments/` = 3 s clips cut by
BirdNET, `clips_1s/` = the 1 s training clips (listed in `manifest_1s.csv`).

**Not in git** because of size: the audio (`Dataset/raw`, `Dataset/Segments`, `Dataset/clips_1s`), the features file
and the trained weights. Ask Jaimin for them.

Order to read: `Dataset/Dataset README.md` -> `Model/Model README.md` -> `Training/Training README.md` ->
`Inference/Inference README.md`.
