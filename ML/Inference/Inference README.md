###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# Inference (not started on the chip)

Nothing runs on the chip yet. Plan for getting the final model (BirdTCN, `ML/Model/weights/seed0.pt`, see
`ML/Model/Model README.md`) onto the nRF52805:

1. **Quantize to int8.** `int8_check.py` simulates it in Python: batch norm folded into the conv weights, 8-bit weights
   (one scale per output channel), 8-bit feature maps with ranges measured on 6,000 train clips. Result on the clean
   3 s test: **0.901 in 8-bit vs 0.901 in float** (0.903 in an earlier run with a different set of calibration clips),
   so 8-bit costs nothing. This is a simulation, not the real C code, so check again after export.
   ```
   python ML/Inference/int8_check.py
   ```
2. **Fold batch norm** into the conv weights (as in `int8_check.py`).
3. **Export** the weights, the 13 normalization stds saved with the model, and the 8-bit ranges as C arrays. (The 13
   train means are saved too, but they cancel out once each clip's own mean is removed, so the chip does not need them.)
4. **On the chip, for each 1 s of audio:** June's MFCC up to the 40 mel energies -> **PCEN** per band (alpha 0.8,
   delta 10, r 0.25, T 0.15 s; see the Training README) -> DCT, 13 coefficients -> divide by the train std -> subtract
   each coefficient's own mean over the 98 frames -> model -> softmax -> average 3 consecutive outputs -> log the top
   class and time to the SD card.
5. **Check end to end:** feed the same clips to Python and to the chip, and compare June's C features with the
   Python version and the model outputs with PyTorch (golden vectors).

Budget: peak RAM for the model 6.1 KB of 24 KB, weights about 28 KB of 192 KB flash, 1.29 M multiply-adds per
1 s clip. PCEN adds 40 bands x 98 frames = about 4,000 updates per second, each with a divide and two powers; the
nRF52805 has no floating-point unit, so use fixed point or lookup tables. On the chip the PCEN average carries over from
one second to the next (training started it at each clip's mean), so check that difference with golden vectors.

Open points for the team: the bit width June's integer MFCC will use (the model was trained on high-precision
features; fewer bits may need retraining), PCEN and the "subtract each clip's mean" step after her MFCC, and whether
the decision window stays at 3 s.
