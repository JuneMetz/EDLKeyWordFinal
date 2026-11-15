###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# Figures

Diagrams and result plots for the bird classifier. Unless a row says otherwise, results are for the **final model**:
BirdTCN on June's MFCC with PCEN, trained on train + val + extra iNaturalist recordings, batch 32, distilled from a larger
teacher model, 38 epochs. Test = the official clean test, 3 s decision (1,431 clips), 3 seeds.
**Final model: test 0.901 +/- 0.006 macro accuracy** (simulated 8-bit: 0.901 to 0.903). Weights in `ML/Model/weights/`,
recipe in `ML/Training/train.py`.

| File | Shows |
|---|---|
| `1_pipeline.png` | whole chain: recordings -> BirdNET labels -> 3 s and 1 s clips -> June's MFCC with PCEN -> clip mean removed -> BirdTCN -> 3 s decision |
| `2_architecture.png` | BirdTCN layers, output shapes and weights: 7,616 + 19,712 + 65 + 520 = 27,913 (BN included); peak RAM 6.1 KB in 8-bit |
| `3_progress.png` | test macro accuracy after each step, from the first DS-CNN (0.603) to the final model (0.901). The first five bars used older, smaller test sets, so they do not compare exactly with the rest |
| `4_per_class.png` | final model, accuracy per class, error bars = std over 3 seeds |
| `5_confusion.png` | final model confusion matrix, 3 seeds summed, rows sum to 1. Biggest errors: Oriole -> Catbird 0.09, Oriole -> Robin 0.05, no bird -> Wren 0.05 |
| `6_window_learning.png` | left: final model, accuracy vs decision window (random 1 s slices of the same test recording, all test clips, not only clean): 1 s 0.73, 3 s 0.88, 5 s 0.90, 10 s 0.91. Right: learning curve of the PCEN BirdTCN trained on train only (before the final extras): accuracy keeps rising with more recordings, but no bird stays flat near 0.78 |
| `7_pr_per_class.png` | final model, precision-recall per class (that class vs the rest). AP: Robin 0.98, Oriole 0.90, Blue Jay 0.98, Wren 0.98, Phoebe 0.99, Catbird 0.97, Hawk 0.91, no bird 0.88; mean 0.95. Dotted lines = chance (the class's share of clips). Curves from seed 1, AP = mean +/- std over 3 seeds |
| `8_pr_bird_vs_nobird.png` | final model, any of our 7 birds vs no bird: AP 0.993. The test is 88 % bird clips (dotted chance line), so this view is easy; no bird on its own has AP 0.88 (Figure 7) |
| `9_old_vs_new_architecture.png` | why BirdTCN: the old 3 x 3 filter slides along time and across the 13 coefficients; BirdTCN treats the 13 coefficients as channels and slides along time only (peak RAM 6.1 KB vs 16.1 KB) |
| `10_validation_experiments.png` | every idea tried with BirdTCN, as the change in clean validation accuracy vs its own control run (test not used). Green = kept in the final model. Most ideas land inside the noise band (about +/- 0.01) |
| `11_per_class_old_vs_new.png` | per class on the clean test: old DS-CNN (0.805) vs BirdTCN with log MFCC (0.854) vs final model (0.901); dashed line = 0.90 target |
| `12_nobird_errors.png` | final model, what happens to the 175 clean no-bird test clips: 134 to 145 correct; about half of the mistakes name the bird the recording was made for, which points to calls BirdNET missed (label noise) rather than model errors |

| `13_all_models.png` | every model and setting tried (77), grouped by stage, with its macro accuracy on the 3 s decision. Dark bars = test score, light bars = validation only; outlined = kept in the final model; blue = DS-CNN, orange = BirdTCN, green = final. Test sets changed with the data (130, 390, then 1,280 recordings), so stages 1 to 3 do not compare exactly with the rest. The right panel uses a zoomed axis (0.82 to 0.92) |

Made by local scripts (`make_figs_final.py`, `make_fig_all_models.py`), not in git.
