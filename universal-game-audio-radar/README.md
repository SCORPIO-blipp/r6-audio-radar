# Universal Game Audio Radar

Real-time footstep detection and direction plotting using Windows WASAPI loopback audio and trained ML models.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

## How it works

1. **WASAPI loopback capture** — records the game audio you hear (no mic needed).
2. **Dual-band filtering** — isolates footstep energy in the 150–450 Hz (thud) and 2500–4000 Hz (surface) bands.
3. **ML classification** — an `MLPClassifier` predicts event type, elevation, and surface material.
4. **Direction estimation** — a stereo → 5.1 surround upmix estimates the horizontal angle of each footstep.
5. **Radar plot** — detections appear as dots on a live radar that decay over time.

## Quick start

### Prerequisites

- **Windows 10 or 11** (WASAPI loopback is Windows-only)
- **Python 3.11 or newer** — [python.org](https://python.org) — check **"Add Python to PATH"** during install
- **Git for Windows** — [git-scm.com](https://git-scm.com)

### Install (one time)

```
git clone https://github.com/SCORPIO-blipp/universal-game-audio-radar.git
cd universal-game-audio-radar
```

Then double-click **`install.bat`**.

It will create a virtual environment and install all dependencies automatically. This takes a few minutes on the first run.

### Run

Double-click **`run.bat`** every time you want to launch the radar.

> Trained model files are already included in the repo — no extra setup needed.

---

### Manual setup (PowerShell / command line)

If you prefer the terminal:

```powershell
cd universal-game-audio-radar
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # PowerShell
# or: .venv\Scripts\activate.bat      # Command Prompt
pip install -e .
python -m universal_game_audio_radar.gui
```

> If PowerShell blocks the activation script, run this once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

## Repository layout

```
universal-game-audio-radar/
├── install.bat                         # One-time setup
├── run.bat                             # Daily launcher
├── pyproject.toml                      # Package metadata & dependencies
├── README.md
├── LICENSE
├── .gitignore
└── universal_game_audio_radar/         # Python package
    ├── __init__.py
    ├── gui.py                          # Tkinter GUI launcher
    ├── runner.py                       # Live audio → filter → classify → plot
    ├── classify.py                     # Load models & classify clips/arrays
    ├── features.py                     # MFCC + spectral feature extraction
    ├── train.py                        # Train the ML models from labelled data
    └── models/                         # Trained .pkl files (included)
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
| **mplcyberpunk** | Cyberpunk theme for matplotlib |

## Training your own models

Prepare a `labels.csv` with columns `filename`, `event`, `elevation`, `material` and a folder of corresponding audio clips, then:

```bash
pip install pandas            # only needed for training
python -m universal_game_audio_radar.train \
    --labels path/to/labels.csv \
    --audio-dir PATH/TO/AUDIO
```

The six `.pkl` files will be written to `universal_game_audio_radar/models/`.

## Running without the GUI

Launch the radar plot directly:

```bash
python -m universal_game_audio_radar.runner
```

Classify a single audio clip:

```bash
python -m universal_game_audio_radar.classify path/to/clip.wav
```

## License

MIT — see [LICENSE](LICENSE).
