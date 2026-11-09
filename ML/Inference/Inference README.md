###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# Inference (not started)

Nothing runs on the chip yet. Plan for getting the trained model (BirdTCN, see `ML/Model/Model README.md`) onto
the nRF52805:

1. **Quantize to int8.** A simulation in Python (8-bit weights per output channel, 8-bit feature maps, ranges taken
   from train clips) gave 0.859 on the clean 3 s test vs 0.860 in float, so int8 should cost almost nothing. This is a
   simulation, not the real C code, so check again after export.
2. **Fold batch norm** into the conv weights.
3. **Export** the weights and the 13 normalization stds saved with the model as C arrays.
4. **On the chip, for each 1 s of audio:** June's MFCC (98 x 13) -> subtract each coefficient's own mean over the
   98 frames -> divide by the train std -> model -> softmax -> average 3 consecutive outputs -> log the top class and
   time to the SD card.
5. **Check end to end:** feed the same clips to Python and to the chip, and compare June's C MFCC with the MATLAB
   version and the model outputs with PyTorch (golden vectors).

Budget: peak RAM for the model 6.1 KB of 24 KB, weights about 28 KB of 192 KB flash, 1.29 M multiply-adds per
1 s clip.

Open points for the team: the bit width June's integer MFCC will use (the model was trained on high-precision
features; fewer bits may need retraining), the new "subtract each clip's mean" step after her MFCC, and whether the
decision window stays at 3 s.
