# R6 Audio Radar

Real-time footstep detection and direction plotting for **Rainbow Six Siege** using Windows WASAPI loopback audio and trained ML models.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

## How it works

1. **WASAPI loopback capture** — records the game audio you hear (no mic needed).
2. **Dual-band filtering** — isolates footstep energy in the 150–450 Hz (thud) and 2500–4000 Hz (surface) bands.
3. **ML classification** — an `MLPClassifier` predicts event type, elevation, and surface material.
4. **Direction estimation** — a stereo → 5.1 Pro-Logic upmix estimates the horizontal angle of each footstep.
5. **Radar plot** — detections appear as dots on a live polar radar that decay over time.

## Repository layout

```
r6-audio-radar/
├── pyproject.toml              # Package metadata & dependencies
├── README.md
├── LICENSE
├── .gitignore
└── r6_audio_radar/             # Python package
    ├── __init__.py
    ├── gui.py                  # Tkinter GUI launcher
    ├── runner.py               # Live audio → filter → classify → plot
    ├── classify.py             # Load models & classify clips/arrays
    ├── features.py             # MFCC + spectral feature extraction
    ├── train.py                # Train the ML models from labelled data
    └── models/                 # Trained .pkl files (see below)
        └── README.md
```

## Requirements

| Dependency | Why |
|---|---|
| **Windows 10/11** | WASAPI loopback is a Windows-only API |
| **Python ≥ 3.11** | f-string / typing features |
| **pyaudiowpatch** | WASAPI loopback capture |
| **numpy, scipy** | Signal processing & filtering |
| **matplotlib** | Live radar plot |
| **librosa, soundfile** | Audio loading & MFCC extraction |
| **scikit-learn** | MLP classifier |
| **sv-ttk** | Dark-themed Tkinter widgets |

## Quick start

```bash
# Clone the repo
git clone https://github.com/<your-username>/r6-audio-radar.git
cd r6-audio-radar

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows

# Install in editable mode
pip install -e .

# Launch the GUI
r6-audio-radar
```

> **Note:** You must place trained model files (`.pkl`) in `r6_audio_radar/models/` before the ML classifier will work.  Without them the radar still runs using energy-based detection as a fallback.

## Training models

Prepare a `labels.csv` with columns `filename`, `event`, `elevation`, `material` and a folder of corresponding audio clips, then:

```bash
pip install pandas            # only needed for training
python -m r6_audio_radar.train \
    --labels path/to/labels.csv \
    --audio-dir path/to/audio/
```

The six `.pkl` files will be written to `r6_audio_radar/models/`.

## Running without the GUI

You can launch the radar plot directly:

```bash
python -m r6_audio_radar.runner
```

Or classify a single audio clip:

```bash
python -m r6_audio_radar.classify path/to/clip.wav
```

## License

MIT — see [LICENSE](LICENSE).
