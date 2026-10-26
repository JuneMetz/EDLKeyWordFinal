###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai
### Note
This readme acts as a placeholder as Github does not allow uploading empty folders. The exact substructure of ML is subject to change.

### Bird dataset (audio not in git)
Audio is too big for the repo (about 1.3 GB), so `raw/`, `Segments/` and `clips_1s/` are in `.gitignore`. The audio will be shared in the team Google Drive folder "Birds sounds".

- `raw/<Bird>/`: 130 xeno-canto recordings of our 7 birds (Robin 16, Oriole 19, Blue Jay 15, Wren 22, Phoebe 17, Catbird 17, Hawk 24).
- `birdnet_species_list.txt`: our 7 birds plus 11 BirdNET non-bird classes.
- `Birdnet results/`: BirdNET-Analyzer 2.4 GUI, Batch analysis, custom list, min conf 0.1, overlap 0, 3 s chunks.
- `Birdnet results all species/`: same, but all species.
- `Birdnet results our7 low/`: custom list, min conf 0.05.
- `Segments/<Bird>/`: 3 s clips from the Segments tab (min conf 0.7, mode confidence, max 1000). 1,323 clips after removing near-empty ones by spectrogram check: Robin 215, Oriole 96, Blue Jay 194, Wren 200, Phoebe 248, Catbird 259, Hawk 111.
- `Segments/No bird/`: 601 clips. 3 s chunks with no detection in the all-species run, minus any chunk where one of our 7 scored 0.05 or more. "No bird" means "none of our 7", so other species can be in it.
- `make_clips_1s.py`: turns `Segments/` into `clips_1s/`. Caps each recording at 25 clips per class, splits by recording (about 70/15/15, no recording in two splits), converts to mono 16 kHz, cuts 1 s slices, drops bird slices with no call, normalizes to -20 dBFS RMS.
- `clips_1s/<split>/<label>/`: 4,732 one-second slices, 16 kHz mono PCM16.
- `manifest_1s.csv`: one row per slice (path, label, split, recording, source clip, BirdNET confidence).

1 s slices per split (train / val / test):

| Label | Train | Val | Test |
|---|---|---|---|
| American Robin | 311 | 67 | 66 |
| Baltimore Oriole | 163 | 28 | 34 |
| Blue Jay | 255 | 66 | 66 |
| Carolina Wren | 288 | 73 | 57 |
| Eastern Phoebe | 403 | 82 | 74 |
| Gray Catbird | 463 | 105 | 104 |
| Red-tailed Hawk | 188 | 41 | 42 |
| No bird | 1012 | 267 | 477 |

Note: if Segments is run on the top folder, it can silently skip some bird folders (a folder-name matching bug in BirdNET 2.4). Run it once per bird folder to be safe.
