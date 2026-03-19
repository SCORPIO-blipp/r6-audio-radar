"""MFCC + spectral feature extraction for audio classification."""

import os
import librosa
import numpy as np


def extract_features(audio, sr=22050):
    """Extract features from audio.

    Parameters
    ----------
    audio : str | os.PathLike | np.ndarray
        Path to an audio file (wav/m4a/etc.), or a raw audio array (mono or stereo).
    sr : int
        Target sampling rate when loading from disk or resampling.
    """

    # If passed a file path, load from disk.
    if isinstance(audio, (str, os.PathLike)):
        path = os.fspath(audio)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        y, sr = librosa.load(path, sr=sr, mono=True)
    else:
        # Assume already-loaded audio array
        y = np.asarray(audio, dtype=float)
        if y.ndim == 2:
            # Convert stereo to mono
            y = np.mean(y, axis=1)

    if y.size == 0:
        raise ValueError("Empty audio data")

    # Ensure enough length for spectral features and avoid zero-length pad error
    if y.shape[0] < 2048:
        y = librosa.util.fix_length(y, 2048)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    features = np.vstack([mfcc, spec_centroid, zcr])
    return features.mean(axis=1)
