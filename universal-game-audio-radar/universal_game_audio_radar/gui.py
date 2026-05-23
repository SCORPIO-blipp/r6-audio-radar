"""Tkinter GUI launcher for the Radar"""

import math
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pyaudiowpatch as pyaudio
import sv_ttk

_PRESETS = {
    "low":    0.001,
    "medium": 0.00075,
    "high":   0.0005,
}
# These actually control detection frequency — mapped to UGAR_LOW_ENERGY_THRESHOLD.
# The combined ENERGY_THRESHOLD above is rarely the binding gate; the bandpass
# threshold fires first and at much lower energy.
_BANDPASS_PRESETS = {
    "low":    0.0005,   # only strong footsteps
    "medium": 0.0002,   # moderate — filters most ambient noise
    "high":   0.00005,  # very sensitive (original hardcoded default)
}
_ADVANCED_MIN = 0.0005
_ADVANCED_MAX = 0.001

# Matches runner.py's format_detection() output:
# [HH:MM:SS]  source   DIR ###°   material / elevation    ████░░░░  XX%
_DETECTION_RE = re.compile(
    r'\[(\d{2}:\d{2}:\d{2})\]\s+\w+\s+([LRF])\s+(\d+)°\s+([\w-]+)\s*/\s*([\w-]+?)\s+[█░]+\s+(\d+)%'
)


class AudioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Universal Game Audio Radar")
        self.root.geometry("600x820")
        self.root.resizable(False, False)
        self.process = None
        self.raw_window = None
        self.raw_text = None
        self.raw_queue = queue.Queue(maxsize=500)
        self.raw_thread = None
        self.raw_stop_event = threading.Event()
        self.heatmap_positions = []
        self.eq_device_indexes = []

        sv_ttk.set_theme("dark")

        main_frame = ttk.Frame(root, padding=15)
        main_frame.pack(fill="both", expand=True)

        # =========================
        # Device display
        # =========================
        ttk.Label(main_frame, text="Default WASAPI Loopback Device:").pack(anchor="w")
        self.device_label = ttk.Label(main_frame, text="Detecting...", foreground="gray")
        self.device_label.pack(anchor="w", pady=(2, 8))
        self.populate_loopback_devices()

        # =========================
        # Detection sensitivity
        # =========================
        ttk.Label(main_frame, text="Detection Sensitivity:").pack(anchor="w")
        self.threshold_preset = tk.StringVar(value="medium")
        self.energy_threshold = tk.DoubleVar(value=_PRESETS["medium"])

        preset_frame = ttk.Frame(main_frame)
        preset_frame.pack(anchor="w", pady=(5, 2))
        for label, key in [("Low", "low"), ("Medium", "medium"), ("High", "high")]:
            ttk.Radiobutton(
                preset_frame, text=label, variable=self.threshold_preset,
                value=key, command=self._on_preset_change,
            ).pack(side="left", padx=(0, 18))

        self.advanced_var = tk.BooleanVar()
        ttk.Checkbutton(
            main_frame, text="Advanced", variable=self.advanced_var,
            command=self._on_advanced_toggle,
        ).pack(anchor="w", pady=(4, 0))

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

        # =========================
        # Audio enhancement (EQ / frequency boost)
        # =========================
        ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=(10, 6))
        ttk.Label(main_frame, text="Audio Enhancement (hard-of-hearing):").pack(anchor="w")

        self.eq_enabled = tk.BooleanVar()
        ttk.Checkbutton(
            main_frame,
            text="Boost footstep frequencies (150–450 Hz) to output device",
            variable=self.eq_enabled,
            command=self._on_eq_toggle,
        ).pack(anchor="w", pady=(4, 0))

        # EQ sub-frame — hidden until checkbox is ticked
        self.eq_frame = ttk.Frame(main_frame)
        self.eq_frame.columnconfigure(1, weight=1)

        ttk.Label(self.eq_frame, text="Output device:").grid(
            row=0, column=0, sticky="w", pady=(4, 0)
        )
        self.eq_device_var = tk.StringVar()
        self.eq_device_dropdown = ttk.Combobox(
            self.eq_frame, textvariable=self.eq_device_var, state="readonly", width=34
        )
        self.eq_device_dropdown.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))

        ttk.Label(self.eq_frame, text="Boost level:").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        self.eq_boost = tk.DoubleVar(value=3.0)
        boost_row = ttk.Frame(self.eq_frame)
        boost_row.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))
        ttk.Scale(
            boost_row, from_=1.0, to=6.0, orient="horizontal", variable=self.eq_boost,
            command=lambda v: self.eq_boost.set(round(float(v), 1)),
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(boost_row, textvariable=self.eq_boost, width=4).pack(side="left", padx=(6, 0))

        # =========================
        # Options
        # =========================
        ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=(10, 6))
        ttk.Label(main_frame, text="Options:").pack(anchor="w")
        self.pref4 = tk.BooleanVar()
        ttk.Checkbutton(
            main_frame, text="(DEV) Raw Data",
            variable=self.pref4, command=self._on_raw_toggle,
        ).pack(anchor="w")
        self.force_ml_var = tk.BooleanVar()
        ttk.Checkbutton(
            main_frame,
            text="Always run ML model on every chunk (bypasses energy gate)",
            variable=self.force_ml_var,
        ).pack(anchor="w")

        # =========================
        # Buttons
        # =========================
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 5))
        ttk.Button(
            button_frame, text="START", command=self.start_running_block
        ).pack(side="left", padx=(0, 10))
        ttk.Button(
            button_frame, text="STOP", command=self.stop_running_block
        ).pack(side="left")
        self.heatmap_btn = ttk.Button(
            button_frame, text="Show Heatmap",
            command=self._show_heatmap, state="disabled",
        )
        self.heatmap_btn.pack(side="right")

        # =========================
        # Detection history panel
        # =========================
        ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=(10, 6))
        ttk.Label(main_frame, text="Detection History:").pack(anchor="w")

        history_frame = ttk.Frame(main_frame)
        history_frame.pack(fill="both", expand=True, pady=(4, 0))

        cols = ("time", "dir", "material", "elevation", "conf")
        self.history_tree = ttk.Treeview(
            history_frame, columns=cols, show="headings", height=8
        )
        self.history_tree.heading("time",      text="Time")
        self.history_tree.heading("dir",       text="Direction")
        self.history_tree.heading("material",  text="Material")
        self.history_tree.heading("elevation", text="Elevation")
        self.history_tree.heading("conf",      text="Conf")
        self.history_tree.column("time",      width=72,  anchor="center")
        self.history_tree.column("dir",       width=90,  anchor="center")
        self.history_tree.column("material",  width=100, anchor="center")
        self.history_tree.column("elevation", width=110, anchor="center")
        self.history_tree.column("conf",      width=55,  anchor="center")

        # Direction color tags
        self.history_tree.tag_configure("left",  foreground="#6699ff")
        self.history_tree.tag_configure("front", foreground="#66ff99")
        self.history_tree.tag_configure("right", foreground="#ff6666")

        scrollbar = ttk.Scrollbar(
            history_frame, orient="vertical", command=self.history_tree.yview
        )
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        self.history_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.monitor_process()

        # Start the output-flush loop — always running, not gated on the raw window
        self.root.after(100, self._flush_raw_output)

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
        value = _PRESETS[self.threshold_preset.get()]
        self.energy_threshold.set(value)
        self.threshold_str.set(str(value))

    def _on_advanced_toggle(self):
        if self.advanced_var.get():
            self.advanced_frame.pack(anchor="w", pady=(4, 2), fill="x")
        else:
            self.advanced_frame.pack_forget()
            self._on_preset_change()

    def _resolve_threshold(self):
        if self.advanced_var.get():
            try:
                val = float(self.threshold_str.get())
            except ValueError:
                print(f"Invalid threshold '{self.threshold_str.get()}', reverting to preset.")
                self._on_preset_change()
                return self.energy_threshold.get()
            if not (_ADVANCED_MIN <= val <= _ADVANCED_MAX):
                clamped = max(_ADVANCED_MIN, min(_ADVANCED_MAX, val))
                print(f"Threshold {val} out of [{_ADVANCED_MIN}, {_ADVANCED_MAX}], clamping to {clamped}.")
                val = clamped
            self.energy_threshold.set(val)
            self.threshold_str.set(str(val))
        return self.energy_threshold.get()

    # ---------------------------------------------------------
    # EQ / frequency boost controls
    # ---------------------------------------------------------

    def _on_eq_toggle(self):
        if self.eq_enabled.get():
            self.eq_frame.pack(anchor="w", pady=(4, 0), fill="x")
            if not self.eq_device_dropdown["values"]:
                self._populate_output_devices()
        else:
            self.eq_frame.pack_forget()

    def _populate_output_devices(self):
        """Enumerate audio output devices for the EQ output dropdown."""
        try:
            p = pyaudio.PyAudio()
            devices = []
            indexes = []
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev.get("maxOutputChannels", 0) > 0:
                    devices.append(f"{i}: {dev['name']}")
                    indexes.append(i)
            default_idx = None
            try:
                default_idx = p.get_default_output_device_info().get("index")
            except Exception:
                pass
            p.terminate()

            self.eq_device_indexes = indexes
            if devices:
                self.eq_device_dropdown["values"] = devices
                if default_idx is not None and default_idx in indexes:
                    self.eq_device_dropdown.current(indexes.index(default_idx))
                else:
                    self.eq_device_dropdown.current(0)
            else:
                self.eq_device_dropdown["values"] = ["No output devices found"]
                self.eq_device_dropdown.current(0)
        except Exception as e:
            self.eq_device_dropdown["values"] = [f"Error: {e}"]
            self.eq_device_dropdown.current(0)

    # ---------------------------------------------------------
    # Heatmap
    # ---------------------------------------------------------

    def _show_heatmap(self):
        if not self.heatmap_positions:
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.15, 1.15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Session Detection Heatmap", color="white", fontsize=12, pad=8)

        # Radar outline
        theta = np.linspace(np.pi, 0, 200)
        for r in [0.33, 0.66, 1.0]:
            ax.plot(
                np.cos(theta) * r, np.sin(theta) * r,
                color="#444466", linewidth=1 if r < 1.0 else 2,
            )
        ax.plot([-1.0, 1.0], [0, 0], color="#444466", linewidth=1)
        for deg in [-60, -30, 0, 30, 60]:
            rad = np.deg2rad(deg)
            ax.plot([0, np.sin(rad)], [0, np.cos(rad)], color="#444466", linewidth=0.5, alpha=0.5)
        ax.text(0, 1.08, "F", ha="center", va="bottom", fontsize=9, color="white")
        ax.text(-1.08, 0, "L", ha="right",  va="center", fontsize=9, color="white")
        ax.text(1.08, 0,  "R", ha="left",   va="center", fontsize=9, color="white")

        xs = [p[0] for p in self.heatmap_positions]
        ys = [p[1] for p in self.heatmap_positions]

        if len(xs) >= 5:
            hb = ax.hexbin(
                xs, ys, gridsize=12, cmap="hot",
                extent=(-1.15, 1.15, -0.15, 1.15), alpha=0.85,
            )
            fig.colorbar(hb, ax=ax, label="Detections").ax.yaxis.label.set_color("white")
        else:
            ax.scatter(xs, ys, c="#ff4444", s=120, alpha=0.7, edgecolors="white", linewidths=0.5)

        ax.plot(0, 0, marker="^", markersize=10, color="white")
        fig.tight_layout()

        win = tk.Toplevel(self.root)
        win.title("Session Detection Heatmap")
        win.geometry("640x450")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        win.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), win.destroy()))

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
        runner_cmd = [sys.executable, "-m", "universal_game_audio_radar.runner"]
        env = os.environ.copy()
        env["UGAR_ENERGY_THRESHOLD"] = str(threshold)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        # Wire sensitivity preset to the bandpass threshold that actually gates detection.
        # In advanced mode leave the defaults so the user's custom value applies only to
        # the combined threshold (UGAR_ENERGY_THRESHOLD).
        if not self.advanced_var.get():
            preset_key = self.threshold_preset.get()
            low_bp = _BANDPASS_PRESETS[preset_key]
            env["UGAR_LOW_ENERGY_THRESHOLD"] = str(low_bp)
            env["UGAR_HIGH_ENERGY_THRESHOLD"] = str(low_bp * 2)

        if self.eq_enabled.get():
            env["UGAR_EQ_ENABLED"] = "1"
            env["UGAR_EQ_BOOST"] = str(round(self.eq_boost.get(), 1))
            selected = self.eq_device_var.get()
            if selected and ":" in selected:
                env["UGAR_EQ_OUTPUT_DEVICE"] = selected.split(":")[0].strip()

        if self.force_ml_var.get():
            env["UGAR_FORCE_ML"] = "1"

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
                continue
            if not line:
                break
            try:
                self.raw_queue.put_nowait(line)
            except queue.Full:
                pass

    def _flush_raw_output(self):
        """
        Drain raw_queue every 100 ms.
        - Detection lines → parsed into the history treeview and heatmap positions.
        - All lines → forwarded to the Raw Output window if it is open.
        Always reschedules itself so it runs for the lifetime of the GUI.
        """
        pending = []
        while True:
            try:
                pending.append(self.raw_queue.get_nowait())
            except queue.Empty:
                break

        for line in pending:
            # Try to parse as a detection line
            m = _DETECTION_RE.search(line)
            if m:
                ts, direction, degrees, material, elevation, conf_pct = m.groups()

                elev_sym = (
                    "↑" if "above" in elevation.lower()
                    else "↓" if "below" in elevation.lower()
                    else "="
                )
                dir_arrow = "←" if direction == "L" else "→" if direction == "R" else "↑"
                dir_str = f"{dir_arrow} {degrees}°"
                tag = "left" if direction == "L" else "right" if direction == "R" else "front"

                self.history_tree.insert(
                    "", 0,
                    values=(ts, dir_str, material, f"{elev_sym} {elevation}", f"{conf_pct}%"),
                    tags=(tag,),
                )
                # Keep at most 15 rows
                children = self.history_tree.get_children()
                if len(children) > 15:
                    self.history_tree.delete(children[-1])

                # Accumulate position for heatmap
                sign = -1 if direction == "L" else (1 if direction == "R" else 0)
                angle_rad = math.radians(sign * int(degrees))
                self.heatmap_positions.append((math.sin(angle_rad), math.cos(angle_rad)))
                self.heatmap_btn.configure(state="normal")

            # Forward every line to the raw window if it is open
            if self.raw_window is not None and self.raw_text is not None:
                try:
                    self.raw_text.configure(state="normal")
                    self.raw_text.insert("end", line)
                    self.raw_text.see("end")
                    self.raw_text.configure(state="disabled")
                except Exception:
                    pass

        self.root.after(100, self._flush_raw_output)


def main():
    root = tk.Tk()
    AudioGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
