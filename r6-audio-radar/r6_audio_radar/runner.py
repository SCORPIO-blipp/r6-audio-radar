"""Live audio capture → dual-band filtering → ML classification → radar plot."""

import os
import sys
import time
import queue
import threading

import signal

import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# Audio input backend: use WASAPI loopback via pyaudiowpatch (required).
try:
    import pyaudiowpatch as pyaudio
except ModuleNotFoundError as e:
    raise RuntimeError(
        "pyaudiowpatch is required for WASAPI loopback capture ("
        "what you hear / game audio). Install it with: pip install pyaudiowpatch"
    ) from e


# =========================
# AUDIO INPUT
# =========================

class AudioInput:
    def __init__(self, rate, channels, chunk=2048):
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.q = queue.Queue(maxsize=10)
        self.stream = None
        self.p = pyaudio.PyAudio()

    def callback(self, in_data, frame_count, time_info, status):
        audio = np.frombuffer(in_data, dtype=np.float32).reshape(-1, self.channels)
        try:
            self.q.put_nowait(audio)
        except queue.Full:
            pass
        return (None, pyaudio.paContinue)

    def start(self, device_index=None):
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk,
            stream_callback=self.callback,
        )
        self.stream.start_stream()

    def read(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        if pyaudio:
            try:
                self.p.terminate()
            except Exception:
                pass


# =========================
# FILTER SETUP
# =========================

class SOSFilterState:
    """Stateful SOS filter to avoid restart transients between chunks."""

    def __init__(self, sos):
        self.sos = sos
        self.zi = spsig.sosfilt_zi(sos)

    def apply(self, x):
        y, self.zi = spsig.sosfilt(self.sos, x, zi=self.zi)
        return y


def create_bandpass(fs, low=150, high=450):
    """Create a bandpass SOS filter for low/mid footstep 'thud' energy."""
    return spsig.butter(4, [low / (fs / 2), high / (fs / 2)], btype="band", output="sos")


def create_highband(fs, low=2500, high=4000):
    """Create a bandpass SOS filter for high-frequency footstep surface sounds."""
    return spsig.butter(4, [low / (fs / 2), high / (fs / 2)], btype="band", output="sos")


def create_lfe_filter(fs):
    """Create a low-frequency extension SOS filter."""
    return spsig.butter(2, 120 / (fs / 2), btype="low", output="sos")


def apply_filter(chunk, filter_state):
    mono = np.mean(chunk, axis=1)
    return filter_state.apply(mono)


# =========================
# SURROUND UPMIX (STEREO → 5.1)
# =========================

def upmix_stereo_to_surround(chunk, lfe_filter):
    L = chunk[:, 0]
    R = chunk[:, 1]
    mono = (L + R) * 0.5

    center = mono
    lfe = lfe_filter.apply(mono)

    # Use difference signal for surrounds — extracts ambience/reverb
    # This is the actual Dolby Pro Logic technique
    diff = (L - R) * 0.5
    surround_L = diff
    surround_R = -diff

    return np.column_stack([L, R, center, lfe, surround_L, surround_R])


# =========================
# TRANSIENT DETECTION
# =========================

def detect_footstep(filtered, threshold=0.01):
    energy = np.mean(filtered ** 2)
    return energy > threshold, energy


# =========================
# DIRECTION ESTIMATION
# =========================

def estimate_direction(chunk, lfe_filter):
    surround = upmix_stereo_to_surround(chunk, lfe_filter)

    def energy(ch):
        return np.mean(surround[:, ch] ** 2)

    fl, fr = energy(0), energy(1)
    sl, sr = energy(4), energy(5)

    front_ratio = (fr - fl) / (fr + fl + 1e-8)
    rear_ratio = (sr - sl) / (sr + sl + 1e-8)

    # Weighted blend — trust front channel more than derived rear channels
    horizontal_ratio = front_ratio * 0.8 + rear_ratio * 0.2

    # Angle in degrees: -90 (left) to +90 (right)
    angle = max(-90.0, min(90.0, horizontal_ratio * 90.0))
    angle_rad = np.deg2rad(angle)

    # Always return a unit direction vector; size should be based on confidence/energy
    vector = np.array([np.sin(angle_rad), np.cos(angle_rad)])

    # Confidence based on stereo energy balance and total energy magnitude
    total_energy = np.mean(chunk ** 2)
    energy_conf = min(1.0, total_energy * 10)  # heuristic scaling
    confidence = min(1.0, abs(front_ratio) * 0.8 + abs(rear_ratio) * 0.2 + energy_conf * 0.2)

    return angle, vector, confidence


# =========================
# CLASSIFICATION HELPER
# =========================

# Models live inside the package: r6_audio_radar/models/
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

try:
    from r6_audio_radar import classify as classify_mod
except Exception as e:
    print(f"Warning: could not import classify module: {e}")
    classify_mod = None

model_available = False
if classify_mod is not None:
    try:
        classify_mod.load_models(_MODEL_DIR)
        model_available = True
    except Exception as e:
        print(f"Warning: could not load models from {_MODEL_DIR}: {e}")
        model_available = False


def classify_chunk(chunk, rate):
    """Classify a chunk of audio with the ML model.

    Falls back to None (meaning: no model-based decision available).
    """
    if not model_available:
        return None
    if chunk is None or chunk.size == 0:
        return None

    try:
        mono = np.mean(chunk, axis=1)
        return classify_mod.classify_audio(mono, sr=rate)
    except Exception as e:
        print(f"Classify chunk failed: {e}")
        return None


# =========================
# MAIN
# =========================
def format_detection(cls, angle, combined_energy):
    use_model = cls is not None and cls.get("event") == "footsteps"
    conf = float(cls.get("confidence", 0.0)) if use_model else min(1.0, combined_energy / 0.05)
    elev = cls.get("elevation", "?") if use_model else "-"
    mat = cls.get("material", "?") if use_model else "-"
    source = "model" if use_model else "energy"

    dir_str = f"{'R' if angle > 5 else 'L' if angle < -5 else 'F'} {abs(angle):>3.0f}°"
    bar = "█" * int(conf * 8) + "░" * (8 - int(conf * 8))
    ts = time.strftime("%H:%M:%S")

    return f"[{ts}]  {source:<6}  {dir_str}   {mat} / {elev:<8}  {bar}  {conf:.0%}"
    
def main():
    try:
        # WASAPI loopback is required to capture system (game) audio.
        p = pyaudio.PyAudio()
        device = p.get_default_wasapi_loopback()
        rate = int(device["defaultSampleRate"])
        channels = device["maxInputChannels"]
        device_index = device["index"]
        p.terminate()

        audio = AudioInput(rate, channels)
        audio.start(device_index)

        # Stateful filters to avoid reset transients between chunks
        # Low band (thud): ~150-450Hz
        bandpass_low = SOSFilterState(create_bandpass(rate))
        # High band (glass/marble, etc): ~2500-4000Hz
        bandpass_high = SOSFilterState(create_highband(rate))

        lfe_filter_dir = SOSFilterState(create_lfe_filter(rate))

        print(f"Running at {rate}Hz, {channels}ch")

        # Shared state — must be created before anything that references it
        stop_event = threading.Event()
        detections_lock = threading.Lock()
        detections = []
        DECAY_RATE = 0.95
        MAX_AGE = 2.5
        BASE_SIZE = 20
        SIZE_SCALE = 400

        # =========================
        # PLOT SETUP
        # =========================

        plt.ion()
        fig, ax = plt.subplots()
        # If the user closes the plot window, stop the audio loop as well
        fig.canvas.mpl_connect("close_event", lambda event: stop_event.set())
        ax.set_aspect("equal")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(Circle((0, 0), 1, fill=False, linewidth=2))
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)

        # Cardinal labels
        for label, pos in [
            ("F", (0, 1.05)),
            ("B", (0, -1.1)),
            ("L", (-1.1, 0)),
            ("R", (1.08, 0)),
        ]:
            ax.text(*pos, label, ha="center", va="center", fontsize=9)

        scatter = ax.scatter([], [], c="red", alpha=0.6)

        def signal_handler(sig, frame):
            stop_event.set()
            print("\nStopping...")

        signal.signal(signal.SIGINT, signal_handler)

        def audio_worker():
            while not stop_event.is_set():
                chunk = audio.read()
                if chunk is None:
                    time.sleep(0.001)
                    continue

                # Bandpass the audio in the key footstep bands (low thud + high surface)
                mono = np.mean(chunk, axis=1)
                energy_low = np.mean(bandpass_low.apply(mono) ** 2)
                energy_high = np.mean(bandpass_high.apply(mono) ** 2)

                # Combine energies, but allow either band to trigger detection
                combined_energy = max(energy_low, energy_high)
                is_footstep = combined_energy > 0.01
                if not is_footstep:
                    continue

                cls = classify_chunk(chunk, rate)

                # Fallback to energy-based detection if model isn't available
                if cls is None or cls.get("event") != "footsteps":
                    model_conf = min(1.0, combined_energy / 0.05)
                else:
                    model_conf = float(cls.get("confidence", 0.0))

                if chunk.shape[1] >= 2:
                    angle, vector, direction_conf = estimate_direction(chunk, lfe_filter_dir)
                else:
                    angle, vector, direction_conf = 0.0, np.array([0.0, 1.0]), 0.5

                size = BASE_SIZE + (model_conf * SIZE_SCALE) + (direction_conf * 100)
                size = max(5, size)

                now = time.time()
                with detections_lock:
                    detections.append({"pos": vector, "size": size, "time": now})

                print(format_detection(cls, angle, combined_energy))

        worker = threading.Thread(target=audio_worker, daemon=True)
        worker.start()

        while not stop_event.is_set():
            now = time.time()
            with detections_lock:
                # decay and expire old detections
                for d in detections:
                    d["size"] *= DECAY_RATE
                detections[:] = [
                    d for d in detections if now - d["time"] < MAX_AGE and d["size"] > 1
                ]

                if detections:
                    xs = [d["pos"][0] for d in detections]
                    ys = [d["pos"][1] for d in detections]
                    sizes = [d["size"] for d in detections]
                    scatter.set_offsets(np.c_[xs, ys])
                    scatter.set_sizes(sizes)
                else:
                    scatter.set_offsets(np.empty((0, 2)))

            plt.draw()
            plt.pause(0.02)

        stop_event.set()
        worker.join(timeout=1)
        audio.stop()
        plt.close(fig)
        print("Terminated.")

    except Exception as e:
        print(f"Audio setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
