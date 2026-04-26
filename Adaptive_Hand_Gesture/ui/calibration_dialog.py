"""
GUI dialog for initiating calibration.
Uses tkinter for a clean, lightweight interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Optional, Callable

from calibration.gesture_store import AVAILABLE_ACTIONS
from calibration.calibration_manager import CalibrationManager

logger = logging.getLogger(__name__)


class CalibrationDialog:
    """
    Dialog window for configuring and starting a calibration session.
    """

    def __init__(self, parent, gesture_store, on_start: Callable):
        """
        Args:
            parent: Parent tkinter window (can be None)
            gesture_store: GestureStore instance
            on_start: Callback(name, action) when user clicks Start
        """
        self.gesture_store = gesture_store
        self.on_start = on_start
        self._result = None

        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("Calibrate New Gesture")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#0a182a")

        self._build_ui()
        self.root.grab_set()

    def _build_ui(self):
        root = self.root
        pad = {"padx": 15, "pady": 8}

        # Title
        tk.Label(root, text="🖐 Calibrate Custom Gesture",
                 font=("Helvetica", 14, "bold"),
                 bg="#0a182a", fg="#f5f5f5").grid(
            row=0, column=0, columnspan=2, **pad)

        # Gesture name
        tk.Label(root, text="Gesture Name:", bg="#0a182a", fg="#f5f5f5").grid(
            row=1, column=0, sticky="e", padx=15)
        self.name_var = tk.StringVar(value="My Gesture")
        name_entry = tk.Entry(root, textvariable=self.name_var, width=24,
                               bg="#1a2b3c", fg="#f5f5f5", insertbackground="white",
                               relief="flat", highlightthickness=1,
                               highlightcolor="#00aaff")
        name_entry.grid(row=1, column=1, padx=15, pady=8)

        # Action
        tk.Label(root, text="Action:", bg="#0a182a", fg="#f5f5f5").grid(
            row=2, column=0, sticky="e", padx=15)
        self.action_var = tk.StringVar(value=AVAILABLE_ACTIONS[0])
        action_cb = ttk.Combobox(root, textvariable=self.action_var,
                                 values=AVAILABLE_ACTIONS, state="readonly", width=22)
        action_cb.grid(row=2, column=1, padx=15, pady=8)

        # Instructions
        instr = ("Instructions:\n"
                 "1. Click Start Calibration\n"
                 "2. Hold your custom gesture in front of camera\n"
                 "3. Press SPACE when ready\n"
                 "4. Hold still during 3-2-1 countdown")
        tk.Label(root, text=instr, justify="left",
                 bg="#0a182a", fg="#b4befe",
                 font=("Helvetica", 9)).grid(
            row=3, column=0, columnspan=2, padx=15, pady=5)

        # Existing gestures list
        existing = self.gesture_store.get_all()
        if existing:
            tk.Label(root, text="Saved Gestures:", bg="#0a182a", fg="#f5f5f5",
                     font=("Helvetica", 9, "bold")).grid(
                row=4, column=0, columnspan=2, padx=15, sticky="w")
            frame = tk.Frame(root, bg="#0a182a")
            frame.grid(row=5, column=0, columnspan=2, padx=15, pady=5)
            for g in existing:
                row_f = tk.Frame(frame, bg="#1a2b3c")
                row_f.pack(fill="x", pady=2)
                tk.Label(row_f, text=f"{g.name} → {g.action}",
                         bg="#1a2b3c", fg="#f5f5f5", font=("Helvetica", 8),
                         padx=8).pack(side="left")
                tk.Button(row_f, text="✕",
                          bg="#f38ba8", fg="white", relief="flat",
                          activebackground="#f38ba8",
                          command=lambda n=g.name: self._delete(n)).pack(side="right")

        # Buttons
        btn_frame = tk.Frame(root, bg="#0a182a")
        btn_frame.grid(row=6, column=0, columnspan=2, pady=15)
        tk.Button(btn_frame, text="Start Calibration",
                  bg="#00aaff", fg="#0a182a", font=("Helvetica", 10, "bold"),
                  relief="flat", padx=12, pady=6,
                  command=self._start).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel",
                  bg="#313244", fg="#f5f5f5", relief="flat", padx=12, pady=6,
                  command=self.root.destroy).pack(side="left", padx=5)

    def _start(self):
        name = self.name_var.get().strip()
        action = self.action_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a gesture name.", parent=self.root)
            return
        self.root.destroy()
        self.on_start(name, action)

    def _delete(self, name: str):
        if messagebox.askyesno("Delete", f"Delete gesture '{name}'?", parent=self.root):
            self.gesture_store.remove(name)
            # Rebuild
            self.root.destroy()
            # Reopen (simple approach)
            CalibrationDialog(None, self.gesture_store, self.on_start).show()

    def show(self):
        self.root.mainloop()
