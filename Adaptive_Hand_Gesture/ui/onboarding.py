"""
Onboarding wizard — runs once on first launch (or when user resets profile).

Flow:
  Screen 1 — Welcome + camera preview
  Screen 2 — Finger count selection (how many fingers does the user have?)
  Screen 3 — Guided calibration: system asks for each action one by one
  Screen 4 — Done / summary

All drawn with OpenCV in the same camera window — no Tkinter needed.
Result stored in Settings and GestureStore.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, List, Tuple, Callable
from enum import Enum, auto
from dataclasses import dataclass, field

from core.hand_tracker import HandTracker, HandData
from calibration.gesture_store import GestureStore, CustomGesture
from calibration.calibration_manager import SAMPLE_COUNT

logger = logging.getLogger(__name__)

# ── Professional Navy / Dark Blue Palette (BGR) ───────────────────────────────
C = {
    "bg":        (42,  24,  14),   # Deep Navy
    "panel":     (60,  35,  22),   # Lighter Navy
    "border":    (100, 70,  40),   # Steel Blue
    "accent":    (255, 170, 0),    # Bright Azure
    "accent2":   (220, 140, 60),   # Muted Sky Blue
    "warn":      (60,  60,  230),  # Reddish
    "success":   (100, 220, 80),   # Emerald
    "danger":    (80,  80,  200),  # Crimson
    "text":      (245, 245, 245),  # Off White
    "muted":     (180, 160, 150),  # Grey-Blue
    "white":     (255, 255, 255),
    "black":     (20,  12,  6),    # Midnight
    "highlight": (255, 200, 50),   # Light Cyan
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX

# Actions the wizard will calibrate, in order
WIZARD_ACTIONS = [
    ("move_cursor",  "Move cursor",   "Point your finger to move the cursor"),
    ("left_click",   "Left click",    "This gesture will left-click"),
    ("right_click",  "Right click",   "This gesture will right-click"),
    ("scroll",       "Scroll",        "This gesture activates scroll mode"),
    ("drag",         "Drag",          "Hold this to drag items"),
]

# How many actions to calibrate per finger-count tier — always include all for accessibility
FINGER_COUNT_SLOTS = {
    1: ["move_cursor", "left_click", "right_click", "scroll", "drag"],
    2: ["move_cursor", "left_click", "right_click", "scroll", "drag"],
    3: ["move_cursor", "left_click", "right_click", "scroll", "drag"],
    4: ["move_cursor", "left_click", "right_click", "scroll", "drag"],
    5: ["move_cursor", "left_click", "right_click", "scroll", "drag"],
}


class WizardState(Enum):
    WELCOME        = auto()
    FINGER_COUNT   = auto()
    PRE_CALIBRATE  = auto()   # "Get ready for action X"
    COUNTDOWN      = auto()
    RECORDING      = auto()
    SAVED          = auto()
    CALIBRATE_TL   = auto()   # New: Screen bounds Top-Left
    CALIBRATE_BR   = auto()   # New: Screen bounds Bottom-Right
    DONE           = auto()
    ABORTED        = auto()


@dataclass
class WizardSession:
    finger_count: int = 5
    actions_to_calibrate: List[str] = field(default_factory=list)
    current_action_idx: int = 0
    state: WizardState = WizardState.WELCOME
    samples: List[np.ndarray] = field(default_factory=list)
    countdown_start: float = 0.0
    saved_gestures: List[str] = field(default_factory=list)
    status_msg: str = ""
    progress: float = 0.0
    calib_tl: Optional[Tuple[int, int]] = None
    calib_br: Optional[Tuple[int, int]] = None


class OnboardingWizard:
    """
    Full-screen OpenCV onboarding wizard.
    Call run() — it blocks until wizard is complete or aborted.
    Returns the finger_count chosen by user, or None if aborted.
    """

    def __init__(self, cap: cv2.VideoCapture,
                 hand_tracker: HandTracker,
                 gesture_store: GestureStore,
                 settings,
                 window_name: str = "AdaptiveGesture — Setup"):
        self.cap = cap
        self.tracker = hand_tracker
        self.store = gesture_store
        self.settings = settings
        self.window_name = window_name

        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.session = WizardSession()
        self._selected_finger_btn: int = -1   # hover index

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> Optional[int]:
        """
        Run the wizard. Blocks until done or ESC pressed.
        Returns finger_count (1-5) or None if aborted.
        """
        logger.info("Onboarding wizard started.")
        self.session = WizardSession(state=WizardState.WELCOME)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            hand_data = self.tracker.process_frame(frame)

            canvas = self._make_canvas(frame)

            state = self.session.state

            if state == WizardState.WELCOME:
                self._draw_welcome(canvas)
            elif state == WizardState.FINGER_COUNT:
                self._draw_finger_count(canvas)
            elif state == WizardState.PRE_CALIBRATE:
                self._draw_pre_calibrate(canvas, hand_data)
            elif state == WizardState.COUNTDOWN:
                self._draw_countdown(canvas, hand_data)
            elif state == WizardState.RECORDING:
                self._do_recording(canvas, hand_data)
            elif state == WizardState.SAVED:
                self._draw_saved(canvas)
            elif state == WizardState.CALIBRATE_TL:
                self._draw_calibrate_bounds(canvas, hand_data, "TOP-LEFT")
            elif state == WizardState.CALIBRATE_BR:
                self._draw_calibrate_bounds(canvas, hand_data, "BOTTOM-RIGHT")
            elif state == WizardState.DONE:
                self._draw_done(canvas)
                cv2.imshow(self.window_name, canvas)
                cv2.waitKey(2000)
                break
            elif state == WizardState.ABORTED:
                break

            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            result = self._handle_key(key, hand_data)
            if result == "quit":
                self.session.state = WizardState.ABORTED
                break

        logger.info(f"Wizard finished. State={self.session.state.name}, "
                    f"fingers={self.session.finger_count}, "
                    f"gestures={self.session.saved_gestures}")

        if self.session.state == WizardState.ABORTED:
            return None
        return self.session.finger_count

    # ── Canvas helper ─────────────────────────────────────────────────────────

    def _make_canvas(self, frame: np.ndarray) -> np.ndarray:
        """Blend camera feed into dark canvas."""
        canvas = np.full((self.h, self.w, 3), C["bg"], dtype=np.uint8)
        # Small mirrored preview in top-right corner
        ph, pw = self.h // 4, self.w // 4
        preview = cv2.resize(frame, (pw, ph))
        canvas[10:10+ph, self.w-pw-10:self.w-10] = preview
        cv2.rectangle(canvas,
                      (self.w-pw-10, 10),
                      (self.w-10, 10+ph),
                      C["border"], 1)
        return canvas

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _text(self, canvas, text, x, y, size=0.6, color=None,
              bold=False, center=False, max_width=None):
        color = color or C["text"]
        font = FONT_BOLD if bold else FONT
        if center:
            tw = cv2.getTextSize(text, font, size, 1)[0][0]
            x = x - tw // 2
        if max_width:
            # Word-wrap
            words = text.split()
            line, lines = "", []
            for w in words:
                test = (line + " " + w).strip()
                if cv2.getTextSize(test, font, size, 1)[0][0] > max_width:
                    lines.append(line)
                    line = w
                else:
                    line = test
            lines.append(line)
            for i, l in enumerate(lines):
                cv2.putText(canvas, l, (x, y + i * int(size * 28)),
                            font, size, color, 1, cv2.LINE_AA)
            return y + len(lines) * int(size * 28)
        cv2.putText(canvas, text, (x, y), font, size, color, 1, cv2.LINE_AA)
        return y + int(size * 32)

    def _panel(self, canvas, x1, y1, x2, y2, border_color=None, alpha=0.92):
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), C["panel"], -1)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        # Add subtle glow to border
        cv2.rectangle(canvas, (x1, y1), (x2, y2),
                      border_color or C["border"], 2)

    def _button(self, canvas, x, y, w, h, label, active=False,
                sub=None, color=None):
        bc = color or (C["accent"] if active else C["border"])
        bg = C["accent"] if active else C["panel"]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), bg, -1)
        cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), bc, 2 if active else 1)
        tc = C["black"] if active else C["text"]
        
        # Calculate text scale based on label length to avoid overflow
        text_size = 0.52 if len(label) < 10 else 0.45
        
        tw = cv2.getTextSize(label, FONT_BOLD, text_size, 1)[0][0]
        cv2.putText(canvas, label, (x + w//2 - tw//2, y + h//2 + 5),
                    FONT_BOLD, text_size, tc, 1, cv2.LINE_AA)
        if sub:
            sc = C["black"] if active else C["muted"]
            sw = cv2.getTextSize(sub, FONT, 0.35, 1)[0][0]
            cv2.putText(canvas, sub, (x + w//2 - sw//2, y + h - 10),
                        FONT, 0.35, sc, 1, cv2.LINE_AA)

    def _progress_bar(self, canvas, x, y, w, h, value, color=None):
        cv2.rectangle(canvas, (x, y), (x+w, y+h), C["panel"], -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), C["border"], 1)
        fill = int(w * min(1.0, max(0.0, value)))
        if fill > 0:
            cv2.rectangle(canvas, (x+1, y+1),
                          (x+fill-1, y+h-1),
                          color or C["accent"], -1)

    def _step_dots(self, canvas, total, current):
        """Draw step indicator dots at top center."""
        dot_r, gap = 6, 20
        total_w = total * dot_r * 2 + (total - 1) * gap
        start_x = self.w // 2 - total_w // 2
        for i in range(total):
            cx = start_x + i * (dot_r * 2 + gap) + dot_r
            color = C["accent"] if i == current else C["border"]
            cv2.circle(canvas, (cx, 22), dot_r,
                       color, -1 if i == current else 1)

    # ── Screen: Welcome ───────────────────────────────────────────────────────

    def _draw_welcome(self, canvas):
        self._panel(canvas, 60, 50, self.w - 60, self.h - 60)
        self._step_dots(canvas, 3, 0)

        cy = 100
        self._text(canvas, "AdaptiveGesture", self.w//2, cy,
                   size=1.1, bold=True, color=C["accent"], center=True)
        cy += 50
        self._text(canvas, "Personalized gesture control for everyone",
                   self.w//2, cy, size=0.55, color=C["muted"], center=True)

        cy += 60
        self._panel(canvas, 100, cy, self.w-100, cy+130, C["accent2"])
        cy += 20
        self._text(canvas, "This setup takes about 2 minutes.", 120, cy,
                   size=0.55, color=C["text"])
        cy += 30
        self._text(canvas, "You will:", 120, cy, size=0.52, color=C["muted"])
        cy += 26
        for line in ["  1.  Tell us how many fingers you can use",
                     "  2.  Show us your preferred gesture for each action",
                     "  3.  Start controlling your computer hands-free"]:
            self._text(canvas, line, 120, cy, size=0.48, color=C["text"])
            cy += 24

        # Big start button
        bw, bh = 220, 52
        bx = self.w // 2 - bw // 2
        by = self.h - 120
        self._button(canvas, bx, by, bw, bh,
                     "START SETUP", active=True)
        self._text(canvas, "Press  ENTER  or click above",
                   self.w//2, by + bh + 22,
                   size=0.45, color=C["muted"], center=True)
        self._text(canvas, "ESC = skip setup (use defaults)",
                   self.w//2, by + bh + 42,
                   size=0.4, color=C["muted"], center=True)

    # ── Screen: Finger count ──────────────────────────────────────────────────

    def _draw_finger_count(self, canvas):
        self._panel(canvas, 40, 40, self.w - 40, self.h - 40)
        self._step_dots(canvas, 3, 1)

        self._text(canvas, "How many fingers can you use?",
                   self.w//2, 70, size=0.8, bold=True, center=True)
        self._text(canvas, "Include any finger you can move and control",
                   self.w//2, 100, size=0.48, color=C["muted"], center=True)

        # Finger count buttons — 1 through 5
        labels = ["1 FINGER", "2 FINGERS", "3 FINGERS", "4 FINGERS", "5 FINGERS"]
        subs   = ["MINIMAL", "BASIC", "STANDARD", "EXPERT", "FULL"]
        hints  = ["cursor only", "basic control", "standard", "full control", "complete set"]

        bw, bh, gap = 110, 85, 10
        total_w = 5 * bw + 4 * gap
        start_x = self.w // 2 - total_w // 2

        for i in range(5):
            bx = start_x + i * (bw + gap)
            by = 125
            active = (self.session.finger_count == i + 1)
            # Use smaller text size for buttons to avoid overflow
            self._button(canvas, bx, by, bw, bh,
                         labels[i], active=active, sub=subs[i])
            self._text(canvas, hints[i], bx + bw//2, by + bh + 14,
                       size=0.35, color=C["highlight"] if active else C["muted"],
                       center=True)

        # Explanation of selected count
        fc = self.session.finger_count
        slots = FINGER_COUNT_SLOTS.get(fc, [])
        action_labels = {a[0]: a[1] for a in WIZARD_ACTIONS}

        ey = 260
        self._panel(canvas, 50, ey, self.w-50, ey+115, C["accent"], alpha=0.95)
        ey += 20
        self._text(canvas, f"You will calibrate {len(slots)} gesture(s).",
                   self.w//2, ey, size=0.52, bold=True,
                   color=C["highlight"], center=True)
        ey += 24

        action_labels = {a[0]: a[1] for a in WIZARD_ACTIONS}
        slot_texts = [action_labels.get(s, s) for s in slots]
        line = " | ".join(slot_texts) # Use pipe instead of dot to avoid encoding issues (??)
        self._text(canvas, line, self.w//2, ey,
                   size=0.42, color=C["text"], center=True)

        ey += 26
        self._text(canvas, "Show ANY gesture you like for each action.",
                   self.w//2, ey, size=0.40, color=C["muted"], center=True)
        ey += 18
        self._text(canvas, "The system learns YOUR gestures, not assumed ones.",
                   self.w//2, ey, size=0.40, color=C["muted"], center=True)
        ey += 18
        self._text(canvas, "You can add more gestures later via settings.",
                   self.w//2, ey, size=0.38, color=C["muted"], center=True)

        # Confirm button
        bw2, bh2 = 240, 44
        bx2 = self.w // 2 - bw2 // 2
        by2 = self.h - 90
        self._button(canvas, bx2, by2, bw2, bh2,
                     "CONFIRM & CONTINUE", active=True)
        self._text(canvas, "Press ENTER to confirm  |  1-5 keys to select",
                   self.w//2, by2 + bh2 + 20,
                   size=0.38, color=C["muted"], center=True)

    # ── Screen: Pre-calibrate ─────────────────────────────────────────────────

    def _draw_pre_calibrate(self, canvas, hand_data: Optional[HandData]):
        idx   = self.session.current_action_idx
        slots = self.session.actions_to_calibrate
        total = len(slots)

        action_key = slots[idx]
        action_info = next((a for a in WIZARD_ACTIONS if a[0] == action_key),
                           (action_key, action_key, "Perform this gesture"))

        self._panel(canvas, 40, 40, self.w-40, self.h-40)
        self._step_dots(canvas, 3, 2)

        # Step counter
        self._text(canvas, f"Gesture {idx+1} of {total}",
                   self.w//2, 68, size=0.5, color=C["muted"], center=True)

        # Action name
        self._text(canvas, f'Calibrate:  "{action_info[1]}"',
                   self.w//2, 100, size=0.85, bold=True,
                   color=C["accent"], center=True)

        self._text(canvas, action_info[2],
                   self.w//2, 130, size=0.5, color=C["muted"], center=True)

        # Split into two panels: Camera (left), Recommendation (right)
        w_half = self.w // 2
        pan_y, pan_h = 160, 160
        self._panel(canvas, 60, pan_y, w_half - 10, pan_y + pan_h, C["border"])
        self._panel(canvas, w_half + 10, pan_y, self.w - 60, pan_y + pan_h, C["border"])

        # Top labels for panels
        self._text(canvas, "Live Hand", 60 + (w_half - 70)//2, pan_y + 20, size=0.45, color=C["muted"], center=True)
        self._text(canvas, "Recommended Gesture", w_half + 10 + (w_half - 70)//2, pan_y + 20, size=0.45, color=C["accent"], center=True)

        if hand_data:
            # Draw skeleton on panel
            self._draw_mini_skeleton(canvas, hand_data, 60, pan_y + 20, w_half - 70, pan_h - 20)
            fc = hand_data.finger_count
            fu_labels = ["T", "I", "M", "R", "P"]
            for fi, (up, lbl) in enumerate(zip(hand_data.fingers_up, fu_labels)):
                col = C["success"] if up else C["danger"]
                cx = 80 + fi * 35
                cv2.circle(canvas, (cx, pan_y + pan_h - 20), 10, col, -1)
                self._text(canvas, lbl, cx-4, pan_y + pan_h - 16, size=0.35, color=C["black"])
        else:
            self._text(canvas, "Show your hand", 60 + (w_half - 70)//2, pan_y + pan_h//2, size=0.5, color=C["warn"], center=True)

        # Draw recommendation image
        import os
        img_path = f"assets/recommendations/f{self.session.finger_count}/{action_key}.png"
        if os.path.exists(img_path):
            rec_img = cv2.imread(img_path)
            rw = (self.w - 60) - (w_half + 10) - 20
            rh = pan_h - 40
            rec_img = cv2.resize(rec_img, (rw, rh))
            canvas[pan_y+30 : pan_y+30+rh, w_half+20 : w_half+20+rw] = rec_img
        else:
            self._text(canvas, f"Use {self.session.finger_count} fingers", w_half + 10 + (w_half - 70)//2, pan_y + pan_h//2, size=0.5, color=C["muted"], center=True)

        # Instructions
        iy = 340
        self._text(canvas,
                   "Hold the gesture you want to use for this action,",
                   self.w//2, iy, size=0.5, color=C["text"], center=True)
        iy += 26
        self._text(canvas, "then press  SPACE  when ready.",
                   self.w//2, iy, size=0.52, bold=True,
                   color=C["highlight"], center=True)

        # Skip / back buttons
        self._button(canvas, 80, self.h-80, 120, 40, "SKIP", color=C["muted"])
        self._button(canvas, self.w-200, self.h-80, 120, 40,
                     "SPACE = ready", active=bool(hand_data), color=C["accent"])

        # Progress dots for this action
        dot_x_start = self.w//2 - (total * 18)//2
        for di in range(total):
            col = C["accent"] if di < idx else (C["success"] if di == idx else C["border"])
            cv2.circle(canvas, (dot_x_start + di*18, self.h-100), 5,
                       col, -1 if di <= idx else 1)

    # ── Screen: Countdown ─────────────────────────────────────────────────────

    def _draw_countdown(self, canvas, hand_data: Optional[HandData]):
        elapsed = time.time() - self.session.countdown_start
        remaining = max(0.0, 3.0 - elapsed)

        if remaining <= 0:
            self.session.state = WizardState.RECORDING
            self.session.samples = []
            return

        self._panel(canvas, 40, 40, self.w-40, self.h-40)

        idx = self.session.current_action_idx
        slots = self.session.actions_to_calibrate
        action_key = slots[idx]
        action_info = next((a for a in WIZARD_ACTIONS if a[0] == action_key),
                           (action_key, action_key, ""))

        self._text(canvas, f'Recording "{action_info[1]}" in...',
                   self.w//2, 120, size=0.6, color=C["muted"], center=True)

        # Big countdown number
        num = str(int(remaining) + 1)
        scale = 1.0 + (1.0 - (remaining % 1.0)) * 0.4
        tw = cv2.getTextSize(num, FONT_BOLD, scale * 4, 3)[0][0]
        cv2.putText(canvas, num,
                    (self.w//2 - tw//2, self.h//2 + 40),
                    FONT_BOLD, scale * 4,
                    C["accent"], 3, cv2.LINE_AA)

        self._text(canvas, "HOLD YOUR GESTURE STEADY",
                   self.w//2, self.h//2 + 100,
                   size=0.6, bold=True, color=C["warn"], center=True)

        if hand_data:
            self._draw_mini_skeleton(canvas, hand_data, 40, self.h-120, self.w-80, 80)

    # ── Screen: Recording ─────────────────────────────────────────────────────

    def _do_recording(self, canvas, hand_data: Optional[HandData]):
        """Collect samples and auto-advance when done."""
        if hand_data is not None:
            self.session.samples.append(hand_data.landmarks.copy())

        progress = len(self.session.samples) / SAMPLE_COUNT
        self.session.progress = progress

        # Draw
        self._panel(canvas, 40, 40, self.w-40, self.h-40)

        idx = self.session.current_action_idx
        action_key = self.session.actions_to_calibrate[idx]
        action_info = next((a for a in WIZARD_ACTIONS if a[0] == action_key),
                           (action_key, action_key, ""))

        self._text(canvas, f'Recording "{action_info[1]}"',
                   self.w//2, 90, size=0.75, bold=True,
                   color=C["accent"], center=True)
        self._text(canvas, "Hold completely still!",
                   self.w//2, 125, size=0.55, color=C["warn"], center=True)

        # Large progress bar
        bx, by, bw, bh = 80, 160, self.w-160, 28
        self._progress_bar(canvas, bx, by, bw, bh, progress, C["accent"])
        self._text(canvas,
                   f"{len(self.session.samples)}/{SAMPLE_COUNT} samples",
                   self.w//2, by+bh+22, size=0.5,
                   color=C["text"], center=True)

        # Live skeleton
        if hand_data:
            self._draw_mini_skeleton(canvas, hand_data, 60, 220, self.w-120, 180)
        else:
            self._text(canvas, "Hand not visible — please show your hand",
                       self.w//2, 310, size=0.55, color=C["danger"], center=True)

        # Done
        if len(self.session.samples) >= SAMPLE_COUNT:
            self._save_current_gesture()

    # ── Screen: Saved ─────────────────────────────────────────────────────────

    def _draw_saved(self, canvas):
        idx = self.session.current_action_idx - 1
        slots = self.session.actions_to_calibrate
        action_key = slots[idx] if idx >= 0 else ""
        action_info = next((a for a in WIZARD_ACTIONS if a[0] == action_key),
                           (action_key, action_key, ""))

        self._panel(canvas, 80, 100, self.w-80, self.h-100, C["success"])

        # Big tick
        cv2.putText(canvas, "✓", (self.w//2 - 30, self.h//2 - 10),
                    FONT_BOLD, 3.0, C["success"], 3, cv2.LINE_AA)

        self._text(canvas, f'"{action_info[1]}" gesture saved!',
                   self.w//2, self.h//2 + 60,
                   size=0.7, bold=True, color=C["success"], center=True)

        remaining = len(self.session.actions_to_calibrate) - self.session.current_action_idx
        if remaining > 0:
            self._text(canvas, f"{remaining} more gesture{'s' if remaining>1 else ''} to go",
                       self.w//2, self.h//2 + 95,
                       size=0.5, color=C["muted"], center=True)
            self._text(canvas, "Press SPACE or wait...",
                       self.w//2, self.h//2 + 125,
                       size=0.45, color=C["muted"], center=True)
        else:
            self._text(canvas, "All gestures calibrated!",
                       self.w//2, self.h//2 + 95,
                       size=0.55, color=C["accent"], center=True)

        # Auto-advance after 1.5s
        if not hasattr(self, "_saved_at"):
            self._saved_at = time.time()
        if time.time() - self._saved_at > 1.5:
            del self._saved_at
            self._advance_to_next_action()

    # ── Screen: Bounds Calibration ──────────────────────────────────────────

    def _draw_calibrate_bounds(self, canvas, hand_data: Optional[HandData], target: str):
        self._panel(canvas, 40, 40, self.w-40, self.h-40, C["accent2"])
        self._step_dots(canvas, 4, 3) # Moved total to 4 steps

        self._text(canvas, "Screen Bounds Calibration", self.w//2, 80,
                   size=0.9, bold=True, color=C["accent2"], center=True)
        
        self._text(canvas, f"Point to the {target} corner", self.w//2, 120,
                   size=0.7, color=C["text"], center=True)
        self._text(canvas, "of your comfortable movement area", self.w//2, 150,
                   size=0.5, color=C["muted"], center=True)

        # Draw a preview of current point if hand is visible
        if hand_data:
            idx_tip = hand_data.landmarks_px[8]
            display_x = self.w - idx_tip[0] # Mirrored
            display_y = idx_tip[1]
            cv2.circle(canvas, (display_x, display_y), 15, C["accent"], 2)
            cv2.circle(canvas, (display_x, display_y), 5, C["accent"], -1)
            
            # Show previously saved point if any
            if target == "BOTTOM-RIGHT" and self.session.calib_tl:
                tl = self.session.calib_tl
                cv2.circle(canvas, (self.w - tl[0], tl[1]), 8, C["success"], -1)
                cv2.putText(canvas, "Top-Left Saved", (self.w - tl[0] + 10, tl[1]), 
                            FONT, 0.4, C["success"], 1)

        self._text(canvas, "Press  SPACE  key to capture", 
                   self.w//2, self.h - 100, size=0.6, bold=True, color=C["highlight"], center=True)

    # ── Screen: Done ──────────────────────────────────────────────────────────

    def _draw_done(self, canvas):
        self._panel(canvas, 40, 40, self.w-40, self.h-40, C["accent"])

        self._text(canvas, "Setup Complete!", self.w//2, 110,
                   size=1.0, bold=True, color=C["accent"], center=True)

        saved = self.session.saved_gestures
        self._text(canvas, f"{len(saved)} gesture{'s' if len(saved)!=1 else ''} calibrated",
                   self.w//2, 155,
                   size=0.6, color=C["text"], center=True)

        gy = 195
        self._panel(canvas, 80, gy, self.w-80, gy+120, C["border"])
        gy += 20

        action_labels = {a[0]: a[1] for a in WIZARD_ACTIONS}
        for g in saved:
            lbl = action_labels.get(g, g)
            self._text(canvas, f"  ✓  {lbl}",
                       120, gy, size=0.5, color=C["success"])
            gy += 24

        fc = self.session.finger_count
        voice_only = [a[1] for a in WIZARD_ACTIONS
                      if a[0] not in self.session.actions_to_calibrate]
        if voice_only:
            gy += 10
            self._text(canvas, "Voice-only actions:",
                       120, gy, size=0.48, color=C["muted"])
            gy += 22
            for v in voice_only:
                self._text(canvas, f"  🎤  {v}", 120, gy,
                           size=0.46, color=C["muted"])
                gy += 20

        self._text(canvas, "Starting in 2 seconds...",
                   self.w//2, self.h - 70,
                   size=0.55, color=C["muted"], center=True)

    # ── Key handling ──────────────────────────────────────────────────────────

    def _handle_key(self, key: int, hand_data: Optional[HandData]) -> Optional[str]:
        if key == 27:   # ESC
            return "quit"

        s = self.session.state

        if s == WizardState.WELCOME:
            if key in (13, 32):   # ENTER or SPACE
                self.session.state = WizardState.FINGER_COUNT

        elif s == WizardState.FINGER_COUNT:
            if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                self.session.finger_count = key - ord("0")
            if key in (13, 32):   # confirm
                self._start_calibration_phase()

        elif s == WizardState.PRE_CALIBRATE:
            if key == 32:   # SPACE
                if hand_data is not None:
                    self.session.state = WizardState.COUNTDOWN
                    self.session.countdown_start = time.time()
            elif key == ord("s"):   # skip this action
                self._skip_current_action()

        elif s == WizardState.SAVED:
            if key == 32:
                if hasattr(self, "_saved_at"):
                    del self._saved_at
                self._advance_to_next_action()
        
        elif s == WizardState.CALIBRATE_TL:
            if key == 32 and hand_data:
                pt = hand_data.landmarks_px[8]
                self.session.calib_tl = (int(pt[0]), int(pt[1]))
                self.session.state = WizardState.CALIBRATE_BR
                logger.info(f"Calibrated TL: {self.session.calib_tl}")
        
        elif s == WizardState.CALIBRATE_BR:
            if key == 32 and hand_data:
                pt = hand_data.landmarks_px[8]
                self.session.calib_br = (int(pt[0]), int(pt[1]))
                self._finalize_bounds()
                self.session.state = WizardState.DONE
                logger.info(f"Calibrated BR: {self.session.calib_br}")
        
        return None

    # ── Logic ─────────────────────────────────────────────────────────────────

    def _start_calibration_phase(self):
        fc = self.session.finger_count
        self.session.actions_to_calibrate = list(FINGER_COUNT_SLOTS.get(fc, []))
        self.session.current_action_idx = 0
        self.session.state = WizardState.PRE_CALIBRATE
        logger.info(f"Calibration phase: {self.session.actions_to_calibrate}")

    def _save_current_gesture(self):
        idx = self.session.current_action_idx
        action_key = self.session.actions_to_calibrate[idx]
        action_info = next((a for a in WIZARD_ACTIONS if a[0] == action_key),
                           (action_key, action_key, ""))

        if not self.session.samples:
            logger.warning("No samples to save.")
            return

        avg = np.mean(self.session.samples, axis=0)
        gesture = CustomGesture(
            name=action_info[1],
            action=action_key,
            template=avg.tolist(),
            description=f"Calibrated by user — {action_info[2]}",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.store.add(gesture)
        self.session.saved_gestures.append(action_key)
        self.session.current_action_idx += 1
        self.session.state = WizardState.SAVED
        logger.info(f"Saved gesture: {gesture.name}")

    def _skip_current_action(self):
        self.session.current_action_idx += 1
        self._advance_to_next_action()

    def _advance_to_next_action(self):
        idx = self.session.current_action_idx
        total = len(self.session.actions_to_calibrate)
        if idx >= total:
            # Instead of DONE, go to bounds calibration
            self.session.state = WizardState.CALIBRATE_TL
        else:
            self.session.samples = []
            self.session.state = WizardState.PRE_CALIBRATE

    def _finalize_bounds(self):
        """Save calibrated bounds to settings."""
        tl = self.session.calib_tl
        br = self.session.calib_br
        if tl and br:
            # Cast to standard int to avoid JSON serialization issues with np.int32
            x1, x2 = int(min(tl[0], br[0])), int(max(tl[0], br[0]))
            y1, y2 = int(min(tl[1], br[1])), int(max(tl[1], br[1]))
            self.settings.calibration_bounds = (x1, y1, x2, y2)
            self.settings.save()
            logger.info(f"Bounds finalized and saved: {self.settings.calibration_bounds}")

    # ── Mini skeleton drawing ─────────────────────────────────────────────────

    def _draw_mini_skeleton(self, canvas, hd: HandData,
                             panel_x, panel_y, panel_w, panel_h):
        """Draw hand skeleton scaled into a panel region."""
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17),
        ]
        lm = hd.landmarks[:, :2].copy()
        # Normalize to panel
        lm[:, 0] = (lm[:, 0] - lm[:, 0].min())
        lm[:, 1] = (lm[:, 1] - lm[:, 1].min())
        rng_x = lm[:, 0].max() + 1e-6
        rng_y = lm[:, 1].max() + 1e-6
        scale = min(panel_w * 0.7 / rng_x, panel_h * 0.7 / rng_y)
        lm[:, 0] = lm[:, 0] * scale + panel_x + panel_w * 0.15
        lm[:, 1] = lm[:, 1] * scale + panel_y + panel_h * 0.15
        lm = lm.astype(np.int32)

        for a, b in CONNECTIONS:
            cv2.line(canvas, tuple(lm[a]), tuple(lm[b]),
                     C["border"], 1, cv2.LINE_AA)
        tips = {4, 8, 12, 16, 20}
        for i, (x, y) in enumerate(lm):
            col = C["accent"] if i in tips else C["text"]
            cv2.circle(canvas, (x, y), 4 if i in tips else 3,
                       col, -1)