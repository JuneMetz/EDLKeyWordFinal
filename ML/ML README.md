###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# ML: bird classifier

Every second of audio becomes June's MFCC grid (98 x 13); a small DS-CNN (4,041 weights) gives 8 scores
(7 birds + no bird); the device averages them over 3 s and logs the top class.

| Folder | What is there |
|---|---|
| `Dataset/` | how the labelled 1 s clips were built (xeno-canto + BirdNET), `make_clips_1s.py`, manifest, BirdNET results |
| `Model/` | `model.py`, the DS-CNN |
| `Training/` | `train.py` (training + scoring), `sweep.py` (try many settings) |
| `Inference/` | plan for running the model on the nRF52805 (not started) |
| `Figures/` | diagrams and result plots (pipeline, architecture, results) |

**Current result:** 0.805 macro accuracy on the 3 s decision (official clean test, 3 seeds), target 0.90.
The weakest classes are Catbird, Wren and Robin.

**Not in git** because of size: the audio (`Dataset/raw`, `Dataset/Segments`, `Dataset/clips_1s`), the features file
and the trained weights. Ask Jaimin for them.

Order to read: `Dataset/Dataset README.md` -> `Model/Model README.md` -> `Training/Training README.md` ->
`Inference/Inference README.md`.
