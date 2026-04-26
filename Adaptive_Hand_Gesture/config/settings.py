"""
Configuration and settings management.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List

CONFIG_DIR   = Path.home() / ".adaptive_gesture"
CONFIG_FILE  = CONFIG_DIR / "settings.json"
GESTURES_FILE = CONFIG_DIR / "gestures.json"
PROFILE_FILE = CONFIG_DIR / "profile.json"
LOG_DIR      = CONFIG_DIR / "logs"


@dataclass
class Settings:
    # Camera
    camera_index: int = 0
    frame_width:  int = 640
    frame_height: int = 480
    fps:          int = 30

    # Hand tracking
    max_hands:              int   = 1
    detection_confidence:   float = 0.5
    tracking_confidence:    float = 0.5

    # Cursor
    smoothing_factor:   float = 0.55    # slightly more smoothing out of the box
    movement_threshold: float = 2.0     # lower: small intentional moves register
    screen_margin:      int   = 40
    cursor_speed:       float = 1.0     # 1.0 = no extra scaling
    calibration_bounds: tuple = None    # (x1, y1, x2, y2) in frame pixels

    # Gesture recognition
    gesture_cooldown_ms:    int   = 500      # ms between repeated actions
    gesture_confirm_ms:     int   = 400      # Reduced to 400ms for faster response
    gesture_confirm_frames: int   = 8        # kept for legacy / settings dialog
    finger_bend_threshold:  float = 0.55
    pinch_threshold:        float = 0.06

    # Drag
    drag_hold_frames: int = 8

    # Scroll
    scroll_speed:     int   = 3
    scroll_threshold: float = 0.015

    # Dwell-click
    dwell_click_enabled: bool  = True
    dwell_click_ms:      int   = 900    # ms of stillness before click fires
    dwell_radius_px:     int   = 18     # pixel radius for "still"

    # Voice
    voice_enabled:      bool  = True
    voice_language:     str   = "en-US"
    voice_timeout:      float = 5.0
    voice_phrase_limit: float = 3.0

    # Onboarding / profile
    onboarding_complete: bool      = False
    user_finger_count:   int       = 5
    active_slots:        List[str] = field(default_factory=lambda: [
        "move_cursor", "left_click", "right_click", "scroll", "drag"
    ])

    # UI
    show_landmarks:    bool  = True
    show_fps:          bool  = True
    show_gesture_name: bool  = True
    show_cursor_zone:  bool  = True
    show_dwell_arc:    bool  = True
    overlay_opacity:   float = 0.8
    ui_theme:          str   = "dark"

    # Debug
    debug_mode: bool = False

    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "Settings":
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)
                s = cls()
                for k, v in data.items():
                    if hasattr(s, k):
                        setattr(s, k, v)
                return s
            except Exception:
                pass
        return cls()

    def ensure_dirs(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def reset_profile(self):
        """Clear onboarding so wizard runs again next launch."""
        self.onboarding_complete = False
        self.user_finger_count   = 5
        self.active_slots        = ["move_cursor", "left_click",
                                    "right_click", "scroll", "drag"]
        self.save()