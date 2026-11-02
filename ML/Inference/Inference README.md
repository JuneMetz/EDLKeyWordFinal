###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# Inference (not started)

Nothing runs on the chip yet. Plan for getting the trained model onto the nRF52805:

1. **Quantize to int8** and check the accuracy again on the same test clips (the 24 KB RAM budget assumes int8).
2. **Fold batch norm** into the conv weights.
3. **Export** the weights, plus the 13 normalization means and stds saved with the model, as C arrays.
4. **On the chip:** June's MFCC on each 1 s of audio -> normalize -> model -> softmax -> average 3 consecutive
   outputs -> log the top class and time to the SD card.
5. **Check end to end:** feed the same clips to Python and to the chip, and compare June's C MFCC with the MATLAB
   version and the model outputs with PyTorch (golden vectors).

Open points for the team: the bit width June's integer MFCC will use (the model was trained on high-precision
features; fewer bits may need retraining), and whether the decision window stays at 3 s (5 s gives about +0.03).
