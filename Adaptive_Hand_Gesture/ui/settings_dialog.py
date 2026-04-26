"""
Settings configuration dialog.
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class SettingsDialog:
    """GUI dialog for adjusting system settings."""

    def __init__(self, parent, settings, on_apply: Callable):
        self.settings = settings
        self.on_apply = on_apply

        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("Settings")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#0a182a")

        self._vars = {}
        self._build_ui()
        self.root.grab_set()

    def _slider_row(self, parent, row, label, key, from_, to, resolution=0.01):
        tk.Label(parent, text=label, bg="#0a182a", fg="#f5f5f5",
                 width=22, anchor="w").grid(row=row, column=0, padx=15, pady=4)
        var = tk.DoubleVar(value=getattr(self.settings, key))
        self._vars[key] = var
        sl = tk.Scale(parent, from_=from_, to=to, resolution=resolution,
                      orient="horizontal", variable=var,
                      bg="#0a182a", fg="#f5f5f5", troughcolor="#1a2b3c",
                      highlightthickness=0, length=180)
        sl.grid(row=row, column=1, padx=5)

    def _checkbox_row(self, parent, row, label, key):
        var = tk.BooleanVar(value=getattr(self.settings, key))
        self._vars[key] = var
        tk.Checkbutton(parent, text=label, variable=var,
                       bg="#0a182a", fg="#f5f5f5",
                       selectcolor="#1a2b3c",
                       activebackground="#0a182a").grid(
            row=row, column=0, columnspan=2, padx=15, pady=3, sticky="w")

    def _build_ui(self):
        tk.Label(self.root, text="⚙ Settings",
                 font=("Helvetica", 14, "bold"),
                 bg="#0a182a", fg="#f5f5f5").grid(
            row=0, column=0, columnspan=2, padx=15, pady=10)

        # Cursor
        tk.Label(self.root, text="— Cursor Control —",
                 bg="#0a182a", fg="#00aaff").grid(
            row=1, column=0, columnspan=2)
        self._slider_row(self.root, 2, "Smoothing (0=raw, 1=max)", "smoothing_factor", 0.0, 0.95)
        self._slider_row(self.root, 3, "Cursor Speed", "cursor_speed", 0.5, 3.0)
        self._slider_row(self.root, 4, "Movement Threshold (px)", "movement_threshold", 1, 15, 1)

        # Gesture
        tk.Label(self.root, text="— Gesture Recognition —",
                 bg="#0a182a", fg="#00aaff").grid(
            row=5, column=0, columnspan=2, pady=5)
        self._slider_row(self.root, 6, "Confirm Frames", "gesture_confirm_frames", 1, 15, 1)
        self._slider_row(self.root, 7, "Cooldown (ms)", "gesture_cooldown_ms", 100, 1500, 50)
        self._slider_row(self.root, 8, "Drag Hold Frames", "drag_hold_frames", 3, 20, 1)
        self._slider_row(self.root, 9, "Scroll Speed", "scroll_speed", 1, 10, 1)

        # Display
        tk.Label(self.root, text="— Display —",
                 bg="#0a182a", fg="#00aaff").grid(
            row=10, column=0, columnspan=2, pady=5)
        self._checkbox_row(self.root, 11, "Show hand landmarks", "show_landmarks")
        self._checkbox_row(self.root, 12, "Show FPS", "show_fps")
        self._checkbox_row(self.root, 13, "Show gesture name", "show_gesture_name")
        self._checkbox_row(self.root, 14, "Show cursor zone", "show_cursor_zone")

        # Detection
        tk.Label(self.root, text="— Detection —",
                 bg="#0a182a", fg="#00aaff").grid(
            row=15, column=0, columnspan=2, pady=5)
        self._slider_row(self.root, 16, "Detection Confidence", "detection_confidence", 0.3, 1.0)
        self._slider_row(self.root, 17, "Tracking Confidence", "tracking_confidence", 0.3, 1.0)

        # Buttons
        btn_f = tk.Frame(self.root, bg="#0a182a")
        btn_f.grid(row=18, column=0, columnspan=2, pady=15)
        tk.Button(btn_f, text="Apply & Save",
                  bg="#00aaff", fg="#0a182a", relief="flat", padx=12, pady=6,
                  font=("Helvetica", 10, "bold"),
                  command=self._apply).pack(side="left", padx=5)
        tk.Button(btn_f, text="Cancel",
                  bg="#313244", fg="#f5f5f5", relief="flat", padx=12, pady=6,
                  command=self.root.destroy).pack(side="left", padx=5)

    def _apply(self):
        for key, var in self._vars.items():
            val = var.get()
            if isinstance(val, float) and val == int(val):
                # Check if the setting is meant to be an int
                current = getattr(self.settings, key)
                if isinstance(current, int):
                    val = int(val)
            setattr(self.settings, key, val)
        self.settings.save()
        self.on_apply(self.settings)
        self.root.destroy()
        logger.info("Settings applied and saved.")
