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
import mplcyberpunk


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
    def __init__(self, rate, channels, chunk=8192):
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.q = queue.Queue(maxsize=200)
        self.stream = None
        self.p = pyaudio.PyAudio()
        self.frames_dropped = 0
        self.frames_received = 0

    def callback(self, in_data, frame_count, time_info, status):
        if not in_data:
            return (None, pyaudio.paContinue)

        audio = np.frombuffer(in_data, dtype=np.float32).reshape(-1, self.channels)
        self.frames_received += 1
        try:
            self.q.put_nowait(audio)
        except queue.Full:
            self.frames_dropped += 1

        if self.frames_received % 500 == 0:
            print(f"Runner: callback got {self.frames_received} chunks, dropped {self.frames_dropped}")
        return (None, pyaudio.paContinue)

    def start(self, device_index=None, as_loopback=False):
        options = dict(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk,
            stream_callback=self.callback,
        )

        # pyaudiowpatch exposes loopback devices as regular virtual input devices —
        # opening them as normal input is all that's needed. No special flag required.
        print(f"Runner: opening stream device {device_index}, loopback={as_loopback}, "
              f"rate={self.rate}, channels={self.channels}")

        self.stream = self.p.open(**options)
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
# SURROUND UPMIX (STEREO -> 5.1)
# =========================

def upmix_stereo_to_surround(chunk, lfe_filter):
    L = chunk[:, 0]
    R = chunk[:, 1]
    mono = (L + R) * 0.5

    center = mono
    lfe = lfe_filter.apply(mono)

    # Use difference signal for surrounds - extracts ambience/reverb
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

    # Weighted blend - trust front channel more than derived rear channels
    horizontal_ratio = front_ratio * 0.8 + rear_ratio * 0.2

    # Angle in degrees: -90 (left) to +90 (right)
    # Square root curve spreads detections away from centre
    sign = np.sign(horizontal_ratio)
    angle = max(-90.0, min(90.0, sign * (abs(horizontal_ratio) ** 0.25) * 90.0))
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
        import librosa
        mono = np.mean(chunk, axis=1)
        if rate != 22050:
            mono = librosa.resample(mono, orig_sr=rate, target_sr=22050)
        return classify_mod.classify_audio(mono, sr=22050)
    except Exception as e:
        print(f"Classify chunk failed: {e}")
        return None


# =========================
# MAIN
# =========================
def format_detection(cls, angle, combined_energy):
    use_model = cls is not None and cls.get("event") in ("footstep", "footsteps")
    conf = float(cls.get("confidence", 0.0)) if use_model else min(1.0, combined_energy / 0.05)
    elev = cls.get("elevation", "?") if use_model else "-"
    mat = cls.get("material", "?") if use_model else "-"
    source = "model" if use_model else "energy"

    def _safe_text(s):
        t = str(s)
        try:
            return t.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
        except Exception:
            return t.encode("ascii", errors="replace").decode("ascii")

    dir_str = f"{'R' if angle > 5 else 'L' if angle < -5 else 'F'} {abs(angle):>3.0f}°"
    dir_str = _safe_text(dir_str)
    mat = _safe_text(mat)
    elev = _safe_text(elev)

    bar = "█" * int(conf * 8) + "░" * (8 - int(conf * 8))
    ts = time.strftime("%H:%M:%S")

    return f"[{ts}]  {source:<6}  {dir_str}   {mat} / {elev:<8}  {bar}  {conf:.0%}"


def main():
    try:
        # WASAPI loopback is preferred for game audio capture.
        p = pyaudio.PyAudio()
        print("Runner: querying default WASAPI loopback device...")
        try:
            device = p.get_default_wasapi_loopback()
            print(f"Runner: got WASAPI loopback device: {device.get('name')} (index {device.get('index')})")
        except Exception as e:
            print(f"Runner: default WASAPI loopback unavailable ({e}), falling back to default input device.")
            try:
                device = p.get_default_input_device_info()
                print(f"Runner: got default input device: {device.get('name')} (index {device.get('index')})")
            except Exception as e2:
                print(f"Runner: no input device available: {e2}")
                p.terminate()
                raise RuntimeError("No audio input device available") from e2

        rate = int(device.get("defaultSampleRate", 44100))
        channels = int(device.get("maxInputChannels", 1))
        device_index = device.get("index")

        if channels <= 0 or device_index is None:
            print("Runner: device has no input channels, scanning for alternative input device...")
            device_index = None
            for i in range(p.get_device_count()):
                candidate = p.get_device_info_by_index(i)
                if candidate.get("maxInputChannels", 0) > 0:
                    device_index = candidate.get("index")
                    rate = int(candidate.get("defaultSampleRate", rate))
                    channels = int(candidate.get("maxInputChannels", channels))
                    print(f"Runner: using fallback input device {candidate.get('name')} (index {device_index}, channels {channels})")
                    break
            if device_index is None:
                p.terminate()
                raise RuntimeError("No input devices with channels found")

        p.terminate()

        audio = AudioInput(rate, channels)
        device_name = device.get("name", "").lower()
        as_loopback = bool(device.get("is_loopback", False)) or "loopback" in device_name
        print(f"Runner: starting audio input (index={device_index}, rate={rate}, channels={channels}, loopback={as_loopback})")
        audio.start(device_index, as_loopback=as_loopback)

        # Stateful filters - owned by classifier_worker thread only
        bandpass_low = SOSFilterState(create_bandpass(rate))
        bandpass_high = SOSFilterState(create_highband(rate))
        lfe_filter_dir = SOSFilterState(create_lfe_filter(rate))

        print(f"Running at {rate}Hz, {channels}ch")

        stop_event = threading.Event()
        detections_lock = threading.Lock()
        detections = []
        DECAY_RATE = 0.95
        MAX_AGE = 2.5
        BASE_SIZE = 20
        SIZE_SCALE = 400

        # Detection tuning
        ENERGY_THRESHOLD = float(os.getenv("R6_AUDIO_RADAR_ENERGY_THRESHOLD", "0.00005"))
        LOW_ENERGY_THRESHOLD = float(os.getenv("R6_AUDIO_RADAR_LOW_ENERGY_THRESHOLD", "0.00005"))
        HIGH_ENERGY_THRESHOLD = float(os.getenv("R6_AUDIO_RADAR_HIGH_ENERGY_THRESHOLD", "0.0001"))
        LOW_ENERGY_WEIGHT = float(os.getenv("R6_AUDIO_RADAR_LOW_ENERGY_WEIGHT", "1.4"))
        HIGH_ENERGY_WEIGHT = float(os.getenv("R6_AUDIO_RADAR_HIGH_ENERGY_WEIGHT", "0.7"))
        MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("R6_AUDIO_RADAR_MODEL_CONFIDENCE_THRESHOLD", "0.3"))

        SEGMENT_DURATION = float(os.getenv("R6_AUDIO_RADAR_SEGMENT_DURATION", "0.5"))
        SEGMENT_HOP = float(os.getenv("R6_AUDIO_RADAR_SEGMENT_HOP", str(SEGMENT_DURATION)))

        segment_samples = max(1, int(round(rate * SEGMENT_DURATION)))
        hop_samples = max(1, int(round(rate * SEGMENT_HOP)))

        print(
            f"Runner: ENERGY_THRESHOLD={ENERGY_THRESHOLD}, "
            f"LOW_THRESHOLD={LOW_ENERGY_THRESHOLD}, HIGH_THRESHOLD={HIGH_ENERGY_THRESHOLD}, "
            f"LOW_WEIGHT={LOW_ENERGY_WEIGHT}, HIGH_WEIGHT={HIGH_ENERGY_WEIGHT}, "
            f"segment={SEGMENT_DURATION}s ({segment_samples} samples), hop={SEGMENT_HOP}s ({hop_samples} samples)"
        )
        verbose_energy_reports = 80

        # Segment queue: audio_worker produces, classifier_worker consumes.
        # maxsize=10 caps backlog at ~5s — keeps latency bounded under load.
        segment_queue = queue.Queue(maxsize=10)

        # =========================
        # PLOT SETUP
        # =========================
        plt.style.use("cyberpunk")
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.canvas.mpl_connect("close_event", lambda event: stop_event.set())
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.15, 1.15)
        ax.set_xticks([])
        ax.set_yticks([])

        # Semicircle arc rings at 33%, 66%, 100% range
        theta = np.linspace(np.pi, 0, 200)
        for r in [0.33, 0.66, 1.0]:
            ax.plot(np.cos(theta) * r, np.sin(theta) * r, linewidth=1 if r < 1.0 else 2)

        # Baseline and spoke lines at -90, -45, 0, 45, 90 degrees
        ax.plot([-1.0, 1.0], [0, 0], linewidth=1)
        for deg in [-60, -30, 0, 30, 60]:
            rad = np.deg2rad(deg)
            ax.plot([0, np.sin(rad)], [0, np.cos(rad)], linewidth=0.5, alpha=0.5)

        # Labels
        ax.text(0, 1.08, "F", ha="center", va="bottom", fontsize=9)
        ax.text(-1.08, 0, "L", ha="right", va="center", fontsize=9)
        ax.text(1.08, 0, "R", ha="left", va="center", fontsize=9)

        # Player marker at origin
        ax.plot(0, 0, marker="^", markersize=10, color="white")

        scatter = ax.scatter([], [], c="red", alpha=0.6)

        def signal_handler(sig, frame):
            stop_event.set()
            print("\nStopping...")

        signal.signal(signal.SIGINT, signal_handler)

        # =========================
        # AUDIO WORKER
        # Only buffers audio and slices segments. Does a cheap energy pre-check
        # to skip silence, then hands segments to classifier_worker via segment_queue.
        # =========================
        def audio_worker():
            idle_loops = 0
            buffer_parts = []
            buffer_frames = 0

            while not stop_event.is_set():
                audio_worker.counter += 1
                chunk = audio.read()
                if chunk is None:
                    idle_loops += 1
                    if idle_loops % 1000 == 0:
                        print(f"Runner: waiting for audio data... ({idle_loops} empty reads)")
                    time.sleep(0.001)
                    continue
                idle_loops = 0

                buffer_parts.append(chunk)
                buffer_frames += chunk.shape[0]

                if buffer_frames >= segment_samples:
                    data = np.concatenate(buffer_parts, axis=0)

                    while data.shape[0] >= segment_samples:
                        segment = data[:segment_samples]

                        # Debug: print energy of every segment so we can see what is arriving
                        mono_preview = np.mean(segment, axis=1)
                        seg_energy = np.mean(mono_preview ** 2)
                        if audio_worker.counter % 20 == 0:
                            print(f"Runner: [audio] seg_energy={seg_energy:.8f} threshold={ENERGY_THRESHOLD}")
                        try:
                            segment_queue.put_nowait(segment.copy())
                        except queue.Full:
                            pass

                        if hop_samples >= data.shape[0]:
                            data = np.empty((0, segment.shape[1]), dtype=np.float32)
                            break

                        data = data[hop_samples:]

                    if data.shape[0] > 0:
                        buffer_parts = [data]
                        buffer_frames = data.shape[0]
                    else:
                        buffer_parts = []
                        buffer_frames = 0

        audio_worker.counter = 0

        # =========================
        # CLASSIFIER WORKER
        # Pulls segments from segment_queue and does all expensive work:
        # bandpass filtering, ML inference, direction estimation, radar update.
        # Runs in its own thread so it never stalls the audio buffer loop.
        # =========================
        def classifier_worker():
            classifier_worker.counter = 0

            while not stop_event.is_set():
                try:
                    segment = segment_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                classifier_worker.counter += 1

                mono = np.mean(segment, axis=1)
                raw_energy = np.mean(mono ** 2)
                energy_low = np.mean(bandpass_low.apply(mono) ** 2)
                energy_high = np.mean(bandpass_high.apply(mono) ** 2)

                weighted_low = energy_low * LOW_ENERGY_WEIGHT
                weighted_high = energy_high * HIGH_ENERGY_WEIGHT
                combined_energy = max(weighted_low, weighted_high)

                cls = classify_chunk(segment, rate)
                model_detected = False
                if cls is not None:
                    all_probs = cls.get("all_probs", {})
                    gunfire_prob = max(
                        all_probs.get("gunfire", 0.0),
                        all_probs.get("shooting", 0.0),
                    )
                    model_detected = (
                        cls.get("event") in ("footstep", "footsteps") and
                        float(cls.get("confidence", 0.0)) >= MODEL_CONFIDENCE_THRESHOLD and
                        gunfire_prob < 0.3
                    )

                is_footstep = model_detected or (
                    energy_low > LOW_ENERGY_THRESHOLD
                    or energy_high > HIGH_ENERGY_THRESHOLD
                    or combined_energy > ENERGY_THRESHOLD
                )

                if classifier_worker.counter % verbose_energy_reports == 0:
                    print(
                        f"Runner: raw={raw_energy:.6f} low={energy_low:.6f} high={energy_high:.6f} "
                        f"w_low={weighted_low:.6f} w_high={weighted_high:.6f} comb={combined_energy:.6f} "
                        f"model={model_detected} footstep={is_footstep}"
                    )
                    if cls is not None:
                        print(
                            f"Runner: model event={cls.get('event')} confidence={cls.get('confidence'):.3f}"
                        )

                if not is_footstep:
                    continue

                if cls is None or cls.get("event") not in ("footstep", "footsteps"):
                    model_conf = min(1.0, combined_energy / 0.05)
                else:
                    model_conf = float(cls.get("confidence", 0.0))

                if segment.shape[1] >= 2:
                    angle, vector, direction_conf = estimate_direction(segment, lfe_filter_dir)
                else:
                    angle, vector, direction_conf = 0.0, np.array([0.0, 1.0]), 0.5

                size = BASE_SIZE + (model_conf * SIZE_SCALE) + (direction_conf * 100)
                size = max(5, size)

                now = time.time()
                with detections_lock:
                    detections.append({"pos": vector, "size": size, "time": now})

                safe_line = format_detection(cls, angle, combined_energy)
                try:
                    print(safe_line)
                except UnicodeEncodeError:
                    print(safe_line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"))

        worker = threading.Thread(target=audio_worker, daemon=True)
        classifier = threading.Thread(target=classifier_worker, daemon=True)
        worker.start()
        classifier.start()

        while not stop_event.is_set():
            now = time.time()
            with detections_lock:
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
        classifier.join(timeout=2)
        audio.stop()
        plt.close(fig)
        print("Terminated.")

    except Exception as e:
        print(f"Audio setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
