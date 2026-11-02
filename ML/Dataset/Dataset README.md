###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# Bird dataset

Labelled 1 s clips of our 7 birds (American Robin, Baltimore Oriole, Blue Jay, Carolina Wren, Eastern Phoebe,
Gray Catbird, Red-tailed Hawk) plus **no bird** ("none of our 7"), built from public xeno-canto recordings and
labelled with BirdNET.

The audio is too big for git (raw 3.9 GB, Segments 5.3 GB, clips_1s 1.1 GB), so `raw/`, `Segments/` and `clips_1s/`
are in `.gitignore`. Everything else here (scripts, BirdNET results, manifest, recordist list) is in the repo.

## What is in this folder

| Path | What it is | In git |
|---|---|---|
| `raw/<Bird>/` | 1,280 xeno-canto recordings (mp3 / wav, 16 to 96 kHz) | no |
| `birdnet_species_list.txt` | our 7 birds + 11 BirdNET non-bird classes (Dog, Engine, Siren, ...) | yes |
| `Birdnet results/` | BirdNET, our list, min confidence 0.1 (1,280 CSVs) | yes |
| `Birdnet results all species/` | BirdNET, all ~6,000 species, min confidence 0.1 | yes |
| `Birdnet results our7 low/` | BirdNET, our list, min confidence 0.05 | yes |
| `Segments/<Bird>/`, `Segments/No bird/` | 3 s clips cut from the recordings | no |
| `recordists.csv` | XC id -> recordist (from xeno-canto), used to keep each recordist in one split | yes |
| `make_clips_1s.py` | turns `Segments/` into `clips_1s/` + `manifest_1s.csv` | yes |
| `clips_1s/<split>/<label>/` | 34,404 one-second clips, 16 kHz mono PCM16 | no |
| `manifest_1s.csv` | one row per 1 s clip: path, label, split, recording, recording_bird, source_clip, slice, birdnet_conf, margin_db | yes |

## How the data was built

1. **Recordings** (`raw/`). Downloaded from xeno-canto in three rounds:
   - 130 picked by hand and with strict rules (mostly quality A, US/Canada, one per recordist),
   - 260 more: US/Canada, quality A first then B, one per new recordist,
   - 890 more with relaxed rules: US/Canada plus Mexico to Colombia, quality A, B or C, 20 s to 10 min,
     at most 3 recordings per recordist per bird, no imitation / mimicry / begging / juvenile calls.

   Recordings per bird: Robin 289, Oriole 125, Blue Jay 192, Wren 232, Phoebe 99, Catbird 262, Hawk 81.

2. **BirdNET (v2.4) on every recording**, 3 s windows, overlap 0, sensitivity 1, band 0 to 15 kHz, CSV output,
   three runs: our list at 0.1, all species at 0.1, our list at 0.05. The first 130 recordings were run in the GUI
   (Batch analysis tab), the rest with the command-line tool that installs with it
   (`BirdNET-Analyzer.exe <folder> -o <out> --slist birdnet_species_list.txt --min_conf 0.1 --overlap 0 --sensitivity 1 --fmin 0 --fmax 15000 --rtype csv`).

3. **3 s clips** (`Segments/`):
   - **bird clip**: a window where one of our 7 has confidence >= 0.7 (our-list run). The clip goes to that bird's
     folder, so an Oriole heard inside a Wren recording is an Oriole clip.
   - **no-bird clip**: a full 3 s window with **no row at all** in the all-species run (nothing at 0.1) and no row for
     our 7 in the 0.05 run.
   - File names keep the BirdNET confidence, the recording and the time, e.g.
     `0.939_1_XC1135112 - American Robin - Turdus migratorius_12.00s_15.00s.wav`,
     `nobird_XC1002346_Baltimore_Oriole_42.00s_45.00s.wav`.
   - For the first 130 recordings a few near-empty clips were removed after a spectrogram check.

   3 s clips: Robin 2,323, Oriole 453, Blue Jay 1,335, Wren 1,657, Phoebe 1,247, Catbird 2,119, Hawk 373, No bird 4,636.

4. **1 s clips** (`python ML/Dataset/make_clips_1s.py`, about 15 min):
   - at most 25 clips per recording per class (highest BirdNET scores; random for no bird),
   - split about 70 / 15 / 15 % by **recording and recordist**: all recordings of one bird by one recordist go to the
     same split, so a microphone or site is never in both train and test,
   - mono, 16 kHz, cut into 1 s slices (no overlap), bird slices kept only if they hold a call
     (1 to 8 kHz peak >= 10 dB above the clip's quiet level), every slice scaled to -20 dBFS RMS.
   - The script refuses to run if `clips_1s/` exists; move the old folder away first. `--dry-run` only prints counts.

## Result: 1 s clips per split

From 1,244 recordings (the rest gave no usable clip). No recording and no (bird, recordist) group is in two splits.

| Label | Train | Val | Test |
|---|---|---|---|
| American Robin | 3,430 | 719 | 716 |
| Baltimore Oriole | 757 | 180 | 161 |
| Blue Jay | 1,975 | 454 | 403 |
| Carolina Wren | 2,430 | 568 | 525 |
| Eastern Phoebe | 1,919 | 372 | 372 |
| Gray Catbird | 3,604 | 784 | 733 |
| Red-tailed Hawk | 589 | 130 | 118 |
| No bird | 6,939 | 3,577 | 2,949 |
| **Total** | **21,643** | **6,784** | **5,977** |

## Known problems

- **No-bird labels are noisy.** A label audit found many no-bird clips that contain our birds BirdNET missed (faint
  or unusual calls), most of all in recordings where BirdNET never found the recording's own bird, and next to a
  detected call. Training keeps them; scoring uses only the trusted ones (see `ML/Training/Training README.md`).
- **BirdNET 2.4 quirks:** some 44.1 kHz mp3 files fail inside BirdNET ("Input signal length=0"), so convert them to
  wav first. The GUI Segments tab can silently skip bird folders when run on the top folder; run it per bird folder.
- **Imbalance:** Oriole and Hawk have the fewest clips; training uses class weights instead of throwing data away.
