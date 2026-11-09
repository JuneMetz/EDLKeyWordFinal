###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# Figures

Diagrams and result plots for the bird classifier. Figures 1 to 8 are for the old DS-CNN (24 channels, attention
pooling, dilated; test 0.805). Figures 9 to 12 are for the current BirdTCN (test 0.854). All: 3 seeds.

| File | Shows |
|---|---|
| `1_pipeline.png` | whole chain: recordings -> BirdNET labels -> 3 s and 1 s clips -> June's MFCC -> model -> 3 s decision |
| `2_architecture.png` | old DS-CNN layers, output shapes and weights (4,041 total) |
| `3_progress.png` | test macro accuracy (3 s decision) after each step up to the old DS-CNN; test set changed with each data version |
| `4_per_class.png` | old DS-CNN: accuracy per class on the official clean test, error bars over 3 seeds |
| `5_confusion.png` | old DS-CNN: confusion matrix on the official clean test (rows sum to 1) |
| `6_window_learning.png` | old DS-CNN. Left: accuracy vs decision window (1 to 10 s); right: learning curve over share of training recordings |
| `7_pr_per_class.png` | old DS-CNN: precision-recall curve per class (that class vs the rest), 3 s decision, clean test (no-bird clips kept only when their label is trusted), n = 1,431; AP per class, mean 0.86; dotted lines = chance (the class's share of clips, e.g. Hawk 0.04, no bird 0.12). Curves from seed 1, AP = mean +/- std over 3 seeds |
| `8_pr_bird_vs_nobird.png` | old DS-CNN: precision-recall curve for any of our 7 birds vs no bird, same test; AP 0.99, but the test is 88 % bird clips (dotted chance line), so this view is easy: no bird on its own scores AP 0.77 (Figure 7) |
| `9_old_vs_new_architecture.png` | what changed: the old 3 x 3 filter slides along time and across the 13 coefficients; BirdTCN treats the 13 coefficients as channels and slides along time only, so maps are 64 x 49 (peak RAM 6.1 KB vs 16.1 KB) |
| `10_validation_experiments.png` | every setting tried for BirdTCN, scored on clean validation only (test untouched), with peak int8 RAM |
| `11_per_class_old_vs_new.png` | per class on the official clean test, old DS-CNN vs BirdTCN, dashed line = 0.90 target |
| `12_confusion_new.png` | BirdTCN confusion matrix on the official clean test, 3 seeds summed (rows sum to 1) |
