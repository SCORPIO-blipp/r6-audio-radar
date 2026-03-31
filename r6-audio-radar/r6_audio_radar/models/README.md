# Place your trained .pkl model files here some will be included as the files given for training were corrupted and train.py is buggy

This directory holds the serialised scikit-learn models and label-encoders
that the radar uses at runtime:

| File                | Description                          |
| ------------------- | ------------------------------------ |
| `event_model.pkl`   | MLPClassifier for event type         |
| `elev_model.pkl`    | MLPClassifier for elevation          |
| `material_model.pkl`| MLPClassifier for surface material   |
| `enc_event.pkl`     | LabelEncoder for event labels        |
| `enc_elev.pkl`      | LabelEncoder for elevation labels    |
| `enc_material.pkl`  | LabelEncoder for material labels     |

## How to generate them

```bash
pip install pandas          # only needed for training
python -m r6_audio_radar.train \
    --labels path/to/labels.csv \
    --audio-dir path/to/audio/
```

The trained `.pkl` files will be written into this folder automatically.
