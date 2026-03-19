"""Load trained ML models and classify audio clips/arrays."""

import argparse
import os
import pickle

import numpy as np

from r6_audio_radar.features import extract_features

# Default model directory: <package>/models/
_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Global model state
event_model = None
elev_model = None
material_model = None
enc_event = None
enc_elev = None
enc_material = None


def _load_pickle(filename, model_dir=None):
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR
    path = os.path.join(model_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_models(model_dir=None):
    """Load all trained model + encoder pickle files."""
    global event_model, elev_model, material_model
    global enc_event, enc_elev, enc_material

    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR

    event_model = _load_pickle("event_model.pkl", model_dir)
    elev_model = _load_pickle("elev_model.pkl", model_dir)
    material_model = _load_pickle("material_model.pkl", model_dir)
    enc_event = _load_pickle("enc_event.pkl", model_dir)
    enc_elev = _load_pickle("enc_elev.pkl", model_dir)
    enc_material = _load_pickle("enc_material.pkl", model_dir)


def _classify_features(feats):
    """Classify based on precomputed feature vector."""
    event = str(enc_event.inverse_transform(event_model.predict(feats))[0])
    elev = str(enc_elev.inverse_transform(elev_model.predict(feats))[0])
    material = str(enc_material.inverse_transform(material_model.predict(feats))[0])

    event_conf = event_model.predict_proba(feats).max()

    return {
        "event": event,
        "confidence": float(event_conf),
        "elevation": elev,
        "material": material,
    }


def classify_audio(y, sr=22050):
    """Predict event, elevation, material from raw audio data (numpy array)."""
    if event_model is None or enc_event is None:
        raise RuntimeError("Models are not loaded. Call load_models() first.")

    feats = extract_features(y, sr=sr).reshape(1, -1)
    return _classify_features(feats)


def classify_clip(path, sr=22050):
    """Predict event, elevation, material from a single audio clip file."""
    if event_model is None or enc_event is None:
        raise RuntimeError("Models are not loaded. Call load_models() first.")

    feats = extract_features(path, sr=sr).reshape(1, -1)
    return _classify_features(feats)


def _find_valid_clip(clip_path=None):
    if clip_path:
        if os.path.isfile(clip_path):
            try:
                extract_features(clip_path)
                return clip_path
            except Exception:
                pass
        return None

    audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "audio")
    if os.path.isdir(audio_dir):
        for fname in sorted(os.listdir(audio_dir)):
            if fname.lower().endswith(('.wav', '.m4a', '.mp4', '.flac', '.ogg')):
                path = os.path.join(audio_dir, fname)
                try:
                    extract_features(path)
                    return path
                except Exception:
                    continue

    clips_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "clips")
    if os.path.isdir(clips_dir):
        for fname in sorted(os.listdir(clips_dir)):
            if fname.lower().endswith(('.wav', '.m4a', '.mp4', '.flac', '.ogg')):
                path = os.path.join(clips_dir, fname)
                try:
                    extract_features(path)
                    return path
                except Exception:
                    continue

    return None


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify one audio clip using trained SSP models.",
    )
    parser.add_argument("clip", nargs="?", default=None, help="Path to audio clip")
    parser.add_argument(
        "--model-dir",
        default=_DEFAULT_MODEL_DIR,
        help="Directory containing event_model.pkl etc.",
    )
    args = parser.parse_args()

    valid_clip = _find_valid_clip(args.clip) or _find_valid_clip(None)

    if valid_clip is None:
        raise RuntimeError(
            "Could not find a valid audio clip to classify. "
            "Check dataset/audio and dataset/clips.",
        )

    print(f"Using clip: {valid_clip}")
    load_models(args.model_dir)
    result = classify_clip(valid_clip)
    print(result)
