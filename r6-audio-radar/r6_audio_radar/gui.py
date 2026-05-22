"""Tkinter GUI launcher for the Radar"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

import pyaudiowpatch as pyaudio
import sv_ttk

# Threshold preset values: lower value = more sensitive (triggers on quieter sounds).
# Higher value = less sensitive (only loud, clear footsteps trigger it).
_PRESETS = {
    "low":    0.001,    # least sensitive — fewest false positives, may miss quiet steps
    "medium": 0.00075,  # balanced
    "high":   0.0005,   # most sensitive — catches more, including background noise
}
_ADVANCED_MIN = 0.0005
_ADVANCED_MAX = 0.001


class AudioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("R6 Audio Radar")
        self.root.geometry("600x600")
        self.root.resizable(False, False)
        self.process = None
        self.raw_window = None
        self.raw_text = None
        self.raw_queue = queue.Queue(maxsize=500)
        self.raw_thread = None
        self.raw_stop_event = threading.Event()

        sv_ttk.set_theme("dark")

        main_frame = ttk.Frame(root, padding=15)
        main_frame.pack(fill="both", expand=True)

        # =========================
        # Device display (read-only — runner auto-selects the default WASAPI loopback)
        # =========================
        ttk.Label(main_frame, text="Default WASAPI Loopback Device:").pack(anchor="w")
        self.device_label = ttk.Label(main_frame, text="Detecting...", foreground="gray")
        self.device_label.pack(anchor="w", pady=(2, 12))
        self.populate_loopback_devices()

        # =========================
        # Detection sensitivity — Low / Medium / High preset
        # =========================
        ttk.Label(main_frame, text="Detection Sensitivity:").pack(anchor="w")

        self.threshold_preset = tk.StringVar(value="medium")
        self.energy_threshold = tk.DoubleVar(value=_PRESETS["medium"])

        preset_frame = ttk.Frame(main_frame)
        preset_frame.pack(anchor="w", pady=(5, 2))
        for label, key in [("Low", "low"), ("Medium", "medium"), ("High", "high")]:
            ttk.Radiobutton(
                preset_frame,
                text=label,
                variable=self.threshold_preset,
                value=key,
                command=self._on_preset_change,
            ).pack(side="left", padx=(0, 18))

        # =========================
        # Advanced threshold entry (hidden until "Advanced" is checked)
        # =========================
        self.advanced_var = tk.BooleanVar()
        ttk.Checkbutton(
            main_frame,
            text="Advanced",
            variable=self.advanced_var,
            command=self._on_advanced_toggle,
        ).pack(anchor="w", pady=(6, 0))

        self.advanced_frame = ttk.Frame(main_frame)
        ttk.Label(
            self.advanced_frame,
            text=f"Custom threshold ({_ADVANCED_MIN}–{_ADVANCED_MAX}):",
        ).pack(side="left")
        self.threshold_str = tk.StringVar(value=str(_PRESETS["medium"]))
        self.threshold_entry = ttk.Entry(
            self.advanced_frame, textvariable=self.threshold_str, width=10
        )
        self.threshold_entry.pack(side="left", padx=(6, 0))
        # advanced_frame is not packed until the checkbox is ticked

        # =========================
        # Options
        # =========================
        ttk.Label(main_frame, text="Options:").pack(anchor="w", pady=(14, 0))

        self.pref4 = tk.BooleanVar()
        ttk.Checkbutton(
            main_frame,
            text="(DEV) Raw Data",
            variable=self.pref4,
            command=self._on_raw_toggle,
        ).pack(anchor="w")

        # =========================
        # Buttons
        # =========================
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(14, 5))
        ttk.Button(
            button_frame, text="START", command=self.start_running_block
        ).pack(side="left", padx=(0, 10))
        ttk.Button(
            button_frame, text="STOP", command=self.stop_running_block
        ).pack(side="left")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.monitor_process()

    # ---------------------------------------------------------
    # Device detection
    # ---------------------------------------------------------

    def populate_loopback_devices(self):
        """Find and display the default WASAPI loopback device name."""
        try:
            p = pyaudio.PyAudio()
            try:
                device = p.get_default_wasapi_loopback()
                name = device.get("name", "Unknown device")
            except Exception:
                try:
                    device = p.get_default_input_device_info()
                    name = device.get("name", "Unknown device") + " (fallback input)"
                except Exception:
                    name = "No audio input device found"
            p.terminate()
        except Exception as e:
            name = f"Error detecting device: {e}"
        self.device_label.config(text=name)

    # ---------------------------------------------------------
    # Sensitivity controls
    # ---------------------------------------------------------

    def _on_preset_change(self):
        """Apply the selected Low/Medium/High preset threshold."""
        value = _PRESETS[self.threshold_preset.get()]
        self.energy_threshold.set(value)
        self.threshold_str.set(str(value))

    def _on_advanced_toggle(self):
        """Show or hide the custom threshold entry field."""
        if self.advanced_var.get():
            self.advanced_frame.pack(anchor="w", pady=(4, 2), fill="x")
        else:
            self.advanced_frame.pack_forget()
            self._on_preset_change()

    def _resolve_threshold(self):
        """Return the effective threshold value, validating the advanced entry if active."""
        if self.advanced_var.get():
            try:
                val = float(self.threshold_str.get())
            except ValueError:
                print(
                    f"Invalid threshold value '{self.threshold_str.get()}', "
                    "reverting to preset."
                )
                self._on_preset_change()
                return self.energy_threshold.get()

            if not (_ADVANCED_MIN <= val <= _ADVANCED_MAX):
                clamped = max(_ADVANCED_MIN, min(_ADVANCED_MAX, val))
                print(
                    f"Threshold {val} is outside [{_ADVANCED_MIN}, {_ADVANCED_MAX}], "
                    f"clamping to {clamped}."
                )
                val = clamped

            self.energy_threshold.set(val)
            self.threshold_str.set(str(val))

        return self.energy_threshold.get()

    # ---------------------------------------------------------
    # Process management
    # ---------------------------------------------------------

    def start_running_block(self):
        """Launch the runner as a subprocess."""
        if self.process is not None and self.process.poll() is None:
            print("Runner already running")
            return

        self.raw_stop_event.clear()

        threshold = self._resolve_threshold()
        runner_cmd = [sys.executable, "-m", "r6_audio_radar.runner"]
        env = os.environ.copy()
        env["R6_AUDIO_RADAR_ENERGY_THRESHOLD"] = str(threshold)
        # Force UTF-8 stdout from the runner so Unicode bar chars (█░) survive the pipe.
        # PYTHONUNBUFFERED ensures lines arrive immediately instead of in batches.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        popen_kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        try:
            self.process = subprocess.Popen(runner_cmd, **popen_kwargs)
            print(f"Started runner subprocess: {runner_cmd}")
        except Exception as e:
            print(f"Failed to launch runner module, trying file path fallback: {e}")
            runner_script = os.path.join(os.path.dirname(__file__), "runner.py")
            try:
                self.process = subprocess.Popen(
                    [sys.executable, runner_script], **popen_kwargs
                )
                print(f"Started runner subprocess by script: {runner_script}")
            except Exception as e2:
                print(f"Error running runner via script fallback: {e2}")
                self.process = None
                return

        self._start_raw_reader()

        if self.pref4.get():
            self._open_raw_window()

    def stop_running_block(self):
        """Stop the runner subprocess."""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            finally:
                self.process = None

        self._stop_raw_reader()
        self._close_raw_window()

    def monitor_process(self):
        """Periodically check whether the subprocess is still alive."""
        if self.process and self.process.poll() is not None:
            self.process = None
        self.root.after(500, self.monitor_process)

    def _on_close(self):
        self.stop_running_block()
        self.root.destroy()

    # ---------------------------------------------------------
    # Raw output console window
    # ---------------------------------------------------------

    def _on_raw_toggle(self):
        if self.pref4.get():
            self._open_raw_window()
        else:
            self._close_raw_window()

    def _open_raw_window(self):
        if self.raw_window is not None:
            return

        self.raw_window = tk.Toplevel(self.root)
        self.raw_window.title("Raw Output")
        self.raw_window.geometry("800x400")

        self.raw_text = tk.Text(self.raw_window, wrap="none", state="disabled")
        self.raw_text.pack(fill="both", expand=True)

        self.raw_window.protocol("WM_DELETE_WINDOW", self._close_raw_window)
        self._flush_raw_output()

    def _close_raw_window(self):
        if self.raw_window is None:
            return
        try:
            self.raw_window.destroy()
        except Exception:
            pass
        self.raw_window = None
        self.raw_text = None
        self.pref4.set(False)

    def _start_raw_reader(self):
        if self.raw_thread is not None and self.raw_thread.is_alive():
            return
        self.raw_stop_event.clear()
        self.raw_thread = threading.Thread(target=self._read_raw_output, daemon=True)
        self.raw_thread.start()

    def _stop_raw_reader(self):
        self.raw_stop_event.set()
        if self.process and self.process.stdout:
            try:
                self.process.stdout.close()
            except Exception:
                pass
        if self.raw_thread is not None:
            self.raw_thread.join(timeout=1)
            self.raw_thread = None

    def _read_raw_output(self):
        if not self.process or self.process.stdout is None:
            return
        while not self.raw_stop_event.is_set():
            try:
                line = self.process.stdout.readline()
            except Exception:
                # Recover from any per-line read error and keep going.
                continue
            if not line:
                # Empty string means EOF — the runner process exited.
                break
            try:
                self.raw_queue.put_nowait(line)
            except queue.Full:
                pass

    def _flush_raw_output(self):
        if self.raw_window is None or self.raw_text is None:
            return
        try:
            while not self.raw_queue.empty():
                line = self.raw_queue.get_nowait()
                self.raw_text.configure(state="normal")
                self.raw_text.insert("end", line)
                self.raw_text.see("end")
                self.raw_text.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._flush_raw_output)


def main():
    root = tk.Tk()
    AudioGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
