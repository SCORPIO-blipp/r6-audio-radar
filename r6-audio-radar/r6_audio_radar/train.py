"""Train MLPClassifier models for footstep event, elevation, and material."""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from r6_audio_radar.features import extract_features

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def train(labels_csv, audio_dir, output_dir=None):
    """Train and save models + encoders.

    Parameters
    ----------
    labels_csv : str
        Path to a CSV with columns: filename, event, elevation, material.
    audio_dir : str
        Folder containing the audio files referenced in labels_csv.
    output_dir : str | None
        Where to write the .pkl files (defaults to package models/ folder).
    """
    if output_dir is None:
        output_dir = _DEFAULT_MODEL_DIR
    os.makedirs(output_dir, exist_ok=True)

    labels = pd.read_csv(labels_csv)

    X = []
    y_event = []
    y_elev = []
    y_material = []

    for _, row in labels.iterrows():
        audio_path = os.path.join(audio_dir, row.filename)
        try:
            feats = extract_features(audio_path)
        except Exception as e:
            print(f"Skipping {audio_path}: {e}")
            continue

        X.append(feats)
        y_event.append(row.event)
        y_elev.append(row.elevation)
        y_material.append(row.material)

    if len(X) == 0:
        raise RuntimeError(
            "No valid feature vectors were extracted. Check your dataset paths and audio files."
        )

    X = np.array(X)

    # Encode labels
    enc_event = LabelEncoder()
    enc_elev = LabelEncoder()
    enc_material = LabelEncoder()

    y_event = enc_event.fit_transform(y_event)
    y_elev = enc_elev.fit_transform(y_elev)
    y_material = enc_material.fit_transform(y_material)

    # Train separate classifiers (multi-head)
    event_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
    elev_model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500)
    material_model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500)

    event_model.fit(X, y_event)
    elev_model.fit(X, y_elev)
    material_model.fit(X, y_material)

    # Save models and encoders
    for obj, name in [
        (event_model, "event_model.pkl"),
        (elev_model, "elev_model.pkl"),
        (material_model, "material_model.pkl"),
        (enc_event, "enc_event.pkl"),
        (enc_elev, "enc_elev.pkl"),
        (enc_material, "enc_material.pkl"),
    ]:
        with open(os.path.join(output_dir, name), "wb") as f:
            pickle.dump(obj, f)

    print(f"Training complete — models saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train R6 Audio Radar ML models.")
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to labels.csv (columns: filename, event, elevation, material)",
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Directory containing the audio files referenced in labels.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_MODEL_DIR,
        help="Where to save .pkl model files (default: package models/ dir)",
    )
    args = parser.parse_args()
    train(args.labels, args.audio_dir, args.output_dir)


if __name__ == "__main__":
    main()
