###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# ML: bird classifier

Every second of audio becomes June's MFCC grid (98 x 13) with PCEN in place of her log step; each clip's own mean is
removed; a small time-only CNN (BirdTCN, 27,913 weights, about 28 KB flash and 6.1 KB peak RAM in 8-bit) gives 8 scores
(7 birds + no bird); the device averages them over 3 s and logs the top class.

| Folder | What is there |
|---|---|
| `Dataset/` | how the labelled 1 s clips were built (xeno-canto + BirdNET), `make_clips_1s.py`, manifests, BirdNET results, the list of extra iNaturalist recordings |
| `Model/` | `model.py` (BirdTCN, and the teacher used only in training), `weights/` (the final model) |
| `Training/` | `train.py`: the 3 steps that make the final model, and the scoring |
| `Inference/` | plan for running the model on the nRF52805, `int8_check.py` (8-bit simulation) |
| `Figures/` | diagrams and result plots, including every model tried (`13_all_models.png`) |

**Result: 0.901** macro accuracy on the 3 s decision (official clean test, 3 seeds; simulated 8-bit 0.901 to 0.903),
up from 0.854 (BirdTCN with log MFCC) and 0.805 (first DS-CNN); target 0.90. Per class: Robin 0.93, Oriole 0.80,
Blue Jay 0.94, Wren 0.93, Phoebe 0.99, Catbird 0.94, Hawk 0.89, No bird 0.80. What made the difference: BirdTCN with
each clip's mean removed, PCEN, extra iNaturalist recordings for Oriole and Hawk, batch 32, distillation from a larger
teacher model, and training on train + val. Details: `Training/Training README.md`.

Data on disk (`Dataset/`, not in git): `raw/` = the 1,280 downloaded recordings, `Segments/` = 3 s clips cut by
BirdNET, `clips_1s/` = the 1 s training clips (listed in `manifest_1s.csv`).

**Not in git** because of size: the audio (`Dataset/raw`, `Dataset/Segments`, `Dataset/clips_1s`, the iNaturalist
audio), the features files and the teacher weights. The feature code (a PyTorch port of June's MFCC) also stays out,
because June owns the recipe. Ask Jaimin for any of them.

Order to read: `Dataset/Dataset README.md` -> `Model/Model README.md` -> `Training/Training README.md` ->
`Inference/Inference README.md`.
