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


class AudioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("R6 Audio Radar")
        self.root.geometry("600x300")
        self.root.resizable(False, False)
        self.process = None
        self.raw_window = None
        self.raw_text = None
        self.raw_queue = queue.Queue(maxsize=500)
        self.raw_thread = None
        self.raw_stop_event = threading.Event()

        # Apply dark theme
        sv_ttk.set_theme("dark")

        # Main frame
        main_frame = ttk.Frame(root, padding=15)
        main_frame.pack(fill="both", expand=True)

        # =========================
        # Loopback Devices Dropdown
        # =========================
        ttk.Label(main_frame, text="Select Loopback Device:").pack(anchor="w")

        self.device_var = tk.StringVar()
        self.device_dropdown = ttk.Combobox(
            main_frame,
            textvariable=self.device_var,
            state="readonly",
        )
        self.device_dropdown.pack(fill="x", pady=(5, 10))

        self.populate_loopback_devices()

        
        # Selectable Preferences 
        
        ttk.Label(main_frame, text="Preferences:").pack(anchor="w")

        self.pref1 = tk.BooleanVar()
        self.pref2 = tk.BooleanVar()
        self.pref3 = tk.BooleanVar()
        self.pref4 = tk.BooleanVar()

        ttk.Checkbutton(main_frame, text="Enable Noise Reduction", variable=self.pref1).pack(
            anchor="w"
        )
        ttk.Checkbutton(main_frame, text="Auto Gain Control", variable=self.pref2).pack(
            anchor="w"
        )
        ttk.Checkbutton(main_frame, text="Low Latency Mode", variable=self.pref3).pack(
            anchor="w"
        )
        ttk.Checkbutton(
            main_frame,
            text="(DEV) Raw Data",
            variable=self.pref4,
            command=self._on_raw_toggle,
        ).pack(anchor="w")

        ttk.Button(main_frame, compound="center", text="START", command=self.start_running_block).pack(
            anchor="w", pady=(10, 5)
        )
        ttk.Button(main_frame, compound="center", text="STOP", command=self.stop_running_block).pack(
            anchor="w"
        )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Monitor process status
        self.monitor_process()

    
    # Device population and finding
    

    def populate_loopback_devices(self):
        """Populate loopback/stereo-mix devices and auto-select the active WASAPI loopback."""
        try:
            p = pyaudio.PyAudio()
            loopback_devices = []
            loopback_indexes = []

            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                dev_name = dev["name"].lower()

                if any(
                    kw in dev_name
                    for kw in ["loopback", "stereo mix", "what u hear", "wave out mix"]
                ):
                    if dev["max_input_channels"] > 0:
                        loopback_devices.append(f"{i}: {dev['name']}")
                        loopback_indexes.append(i)

            active_index = None
            try:
                default_loopback = p.get_default_wasapi_loopback()
                active_index = default_loopback.get("index")
            except Exception:
                active_index = None

            p.terminate()

            if loopback_devices:
                self.device_dropdown["values"] = loopback_devices
                if active_index is not None and active_index in loopback_indexes:
                    self.device_dropdown.current(loopback_indexes.index(active_index))
                else:
                    self.device_dropdown.current(0)
            else:
                self.device_dropdown["values"] = ["No loopback devices found"]
                self.device_dropdown.current(0)

        except Exception as e:
            print(f"Error detecting loopback devices: {e}")
            self.device_dropdown["values"] = ["Error detecting devices"]
            self.device_dropdown.current(0)

    # ---------------------------------------------------------
    # Process management
    # ---------------------------------------------------------

    def start_running_block(self):
        """Launch the runner as a subprocess."""
        if self.process is None or self.process.poll() is not None:
            try:
                self.raw_stop_event.clear()
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "r6_audio_radar.runner"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                self._start_raw_reader()

                if self.pref4.get():
                    self._open_raw_window()

            except Exception as e:
                print(f"Error running runner: {e}")

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
        try:
            for line in self.process.stdout:
                if self.raw_stop_event.is_set():
                    break
                try:
                    self.raw_queue.put_nowait(line)
                except queue.Full:
                    pass
        except Exception:
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
