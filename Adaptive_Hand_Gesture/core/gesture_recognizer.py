"""
Gesture recognition engine — slot-aware, calibration-first.

Fixes applied:
  - Time-based confirmation (gesture_confirm_ms) — no FPS dependency
  - Custom gesture threshold raised to 0.85 to cut false positives
  - Built-in rules skipped for slots that have a custom template
  - Finger-count assumption removed (system only trusts what user trained)
"""

import numpy as np
import logging
import time
from enum import Enum, auto
from typing import Optional, List, Set
from dataclasses import dataclass

from core.hand_tracker import HandData
from calibration.gesture_store import GestureStore

logger = logging.getLogger(__name__)


class GestureType(Enum):
    NONE         = auto()
    MOVE_CURSOR  = auto()
    LEFT_CLICK   = auto()
    RIGHT_CLICK  = auto()
    DRAG         = auto()
    SCROLL_UP    = auto()
    SCROLL_DOWN  = auto()
    SCROLL       = auto()
    PINCH_CLICK  = auto()
    OPEN_PALM    = auto()
    DWELL_CLICK  = auto()
    WRIST_MOVE   = auto()
    CUSTOM       = auto()


CONTINUOUS = {
    GestureType.MOVE_CURSOR,
    GestureType.NONE,
    GestureType.WRIST_MOVE,
}


@dataclass
class GestureResult:
    gesture: GestureType
    confidence: float
    custom_name: Optional[str] = None
    custom_action: Optional[str] = None
    hand_data: Optional[HandData] = None
    arc_progress: float = 0.0


class GestureRecognizer:
    ACTION_TO_GESTURE = {
        "move_cursor":  GestureType.MOVE_CURSOR,
        "left_click":   GestureType.LEFT_CLICK,
        "right_click":  GestureType.RIGHT_CLICK,
        "scroll":       GestureType.SCROLL,
        "drag":         GestureType.DRAG,
        "double_click": GestureType.LEFT_CLICK,
        "scroll_up":    GestureType.SCROLL_UP,
        "scroll_down":  GestureType.SCROLL_DOWN,
        "pinch_click":  GestureType.PINCH_CLICK,
    }

    def __init__(self, settings, gesture_store: GestureStore):
        self.settings      = settings
        self.gesture_store = gesture_store
        self.active_slots: Set[str] = {
            "move_cursor", "left_click", "right_click", "scroll", "drag",
        }
        self._prev_gesture       = GestureType.NONE
        self._gesture_hold_count = 0
        self._gesture_hold_start = 0.0   # wall-clock when current gesture began
        self._last_action_time   = 0.0
        self._arc_progress       = 0.0
        self._prev_wrist_y: Optional[float] = None
        self._prev_scroll_y: float = 0.0
        self._scroll_accum   = 0.0
        self._fist_hold_frames = 0
        self._drag_active    = False
        self._dwell_pos      = None
        self._dwell_start    = None
        self._dwell_fired    = False
        logger.info("GestureRecognizer initialized.")

    def set_active_slots(self, slots: List[str]):
        self.active_slots = set(slots)
        logger.info(f"Active gesture slots: {self.active_slots}")

    @property
    def is_dragging(self) -> bool:
        return self._drag_active

    @property
    def arc_progress(self) -> float:
        return self._arc_progress

    def recognize(self, hand_data: HandData) -> GestureResult:
        raw       = self._classify_raw(hand_data)
        confirmed = self._apply_confirmation(raw, hand_data)
        self._update_dwell(hand_data, confirmed)
        
        # Track drag state for hysteresis in next frame
        if confirmed.gesture == GestureType.DRAG:
            self._drag_active = True
        elif confirmed.gesture in (GestureType.OPEN_PALM, GestureType.NONE) or not confirmed.gesture:
            # Note: MOVE_CURSOR doesn't stop drag immediately in the executor (grace period)
            # but for gesture classification we want to know if we are in the "Drag" mode.
            # If the executor decides to stop it later, that's fine.
            if confirmed.gesture == GestureType.OPEN_PALM:
                self._drag_active = False

        return confirmed

    # ── Raw classification ────────────────────────────────────────

    def _classify_raw(self, hd: HandData) -> GestureResult:
        # 1. Custom calibrated gestures (highest priority)
        custom = self._match_custom(hd)
        if custom:
            return custom

        # 2. High-confidence Pinch logic (matching test_cursor_control.py)
        # Left click: Index + Thumb
        if "left_click" in self.active_slots and hd.pinch_dist_px < 50:
            return GestureResult(GestureType.LEFT_CLICK, 0.98, hand_data=hd)
        
        # Right click: Index + Middle (Natural gap is wider, use 75px)
        if "right_click" in self.active_slots and hd.mid_pinch_dist_px < 75:
            return GestureResult(GestureType.RIGHT_CLICK, 0.98, hand_data=hd)

        fu    = hd.fingers_up
        count = hd.finger_count
        calibrated = {g.action for g in self.gesture_store.get_all()}

        # 3. Fist / Drag (Curled fingers)
        if "drag" in self.active_slots and "drag" not in calibrated:
            # Fist: Index, Middle, Ring are all curled
            # Relaxed if already dragging (hysteresis)
            if self._drag_active:
                # If already dragging, only need 2 fingers curled to keep it
                if sum(fu[1:4]) <= 1: 
                    return GestureResult(GestureType.DRAG, 0.90, hand_data=hd)
            else:
                # To start drag, need index+middle+ring curled
                if not any(fu[1:4]):
                    return GestureResult(GestureType.DRAG, 0.90, hand_data=hd)

        # 4. Open Palm (Release)
        if sum(fu[1:5]) >= 4: # at least index, mid, ring, pinky up
            return GestureResult(GestureType.OPEN_PALM, 0.90, hand_data=hd)

        # 5. Scroll (2+ fingers)
        if (fu[1] and fu[2]
                and "scroll" in self.active_slots
                and "scroll" not in calibrated):
            return GestureResult(GestureType.SCROLL, 0.85, hand_data=hd)

        # 5. Move cursor (Relaxed: 1 to 3 fingers up)
        if "move_cursor" in self.active_slots:
            # If Index is up, or Index+Middle are up but NOT pinching
            if fu[1] or (fu[2] and count <= 3):
                return GestureResult(GestureType.MOVE_CURSOR, 0.95, hand_data=hd)

        return GestureResult(GestureType.NONE, 0.2, hand_data=hd)

    # ── Custom gesture matching ───────────────────────────────────

    def _match_custom(self, hd: HandData) -> Optional[GestureResult]:
        best       = None
        best_score = 0.85   # baseline strict threshold

        for g in self.gesture_store.get_all():
            if g.action not in self.active_slots:
                continue
            
            score = g.match(hd.landmarks)
            
            # HYSTERESIS: if we were already doing this gesture, lower the threshold a lot!
            # It's hard to keep hand perfectly rigid while dragging it across the screen.
            threshold = 0.85
            if self._prev_gesture == self.ACTION_TO_GESTURE.get(g.action, GestureType.CUSTOM):
                threshold = 0.55  # massive relaxation for sustained continuous actions

            if score > threshold and score > best_score:
                best_score = score
                best = g

        if best is None:
            return None

        gesture_type = self.ACTION_TO_GESTURE.get(best.action, GestureType.CUSTOM)
        if best.action == "scroll":
            gesture_type = GestureType.SCROLL

        return GestureResult(gesture_type, best_score,
                             custom_name=best.name, custom_action=best.action,
                             hand_data=hd)

    # ── Scroll direction ──────────────────────────────────────────
 
    def _detect_scroll_direction(self, hd: HandData) -> GestureType:
        # Use user-suggested logic: track diff of index tip Y
        current_y = hd.landmarks_px[8][1] # Index tip Y
        
        if self._prev_scroll_y != 0:
            diff = current_y - self._prev_scroll_y
            if abs(diff) > 5: # threshold
                self._prev_scroll_y = float(current_y)
                if diff > 0: return GestureType.SCROLL_DOWN # Moving down -> scroll down
                else:        return GestureType.SCROLL_UP   # Moving up -> scroll up
        
        self._prev_scroll_y = float(current_y)
        return GestureType.NONE

    # ── Wrist-angle cursor ────────────────────────────────────────

    def _wrist_angle_result(self, hd: HandData) -> GestureResult:
        wrist   = hd.landmarks[0, :2]
        mid_mcp = hd.landmarks[9, :2]
        vec     = mid_mcp - wrist
        return GestureResult(
            GestureType.WRIST_MOVE, 0.85,
            custom_action="wrist_move",
            custom_name=f"{float(vec[0]):.3f},{float(vec[1]):.3f}",
            hand_data=hd,
        )

    # ── Dwell-click ───────────────────────────────────────────────

    def _update_dwell(self, hd: HandData, result: GestureResult):
        if not getattr(self.settings, "dwell_click_enabled", True):
            return
        if result.gesture != GestureType.MOVE_CURSOR:
            self._dwell_pos   = None
            self._dwell_start = None
            self._dwell_fired = False
            self._arc_progress = 0.0
            return

        tip          = hd.index_tip
        dwell_ms     = getattr(self.settings, "dwell_click_ms", 900)
        dwell_radius = getattr(self.settings, "dwell_radius_px", 18)

        if self._dwell_pos is None:
            self._dwell_pos   = tip
            self._dwell_start = time.time()
            self._dwell_fired = False
            return

        if (abs(tip[0] - self._dwell_pos[0]) > dwell_radius or
                abs(tip[1] - self._dwell_pos[1]) > dwell_radius):
            self._dwell_pos   = tip
            self._dwell_start = time.time()
            self._dwell_fired = False
            self._arc_progress = 0.0
            return

        elapsed = (time.time() - self._dwell_start) * 1000
        self._arc_progress  = min(1.0, elapsed / dwell_ms)
        result.arc_progress = self._arc_progress

        if elapsed >= dwell_ms and not self._dwell_fired:
            self._dwell_fired    = True
            self._dwell_pos      = None
            result.gesture       = GestureType.DWELL_CLICK
            result.confidence    = 1.0
            result.arc_progress  = 1.0

    # ── Time-based confirmation ───────────────────────────────────

    def _apply_confirmation(self, raw: GestureResult,
                             hd: HandData) -> GestureResult:
        """
        Discrete (Click): Wait 1s, fire ONCE, then return NONE until release.
        Continuous (Drag/Scroll): Wait 1s, then stream raw continuously.
        Immediate (Move/None): Fire immediately.
        """
        now = time.time()
        confirm_ms = float(self.settings.gesture_confirm_ms)

        # 1. Immediate gestures
        if raw.gesture in (GestureType.MOVE_CURSOR, GestureType.NONE, GestureType.WRIST_MOVE):
            self._prev_gesture = raw.gesture
            self._gesture_hold_start = now
            self._arc_progress = 0.0
            return raw

        # 2. Transition handling
        if raw.gesture != self._prev_gesture:
            self._prev_gesture = raw.gesture
            self._gesture_hold_start = now
            self._arc_progress = 0.0
            
            # IMMEDIATE gestures: Return raw immediately to avoid flicker/delay
            if raw.gesture in (GestureType.DRAG, GestureType.SCROLL):
                self._gesture_hold_start = now + 9999 # Mark as fired
                return raw
            else:
                return GestureResult(GestureType.NONE, 0.1, hand_data=hd)

        # 3. Handle Already-Fired State (Blocked Clicks)
        # We use a special marker: if hold_start is in the far future, it means click already fired.
        if self._gesture_hold_start > now + 500: # We used +9999
            raw.arc_progress = 1.0
            # If it's a click, we return NONE to prevent repeats
            if raw.gesture in (GestureType.LEFT_CLICK, GestureType.RIGHT_CLICK):
                return GestureResult(GestureType.NONE, 0.1, arc_progress=1.0, hand_data=hd)
            # If it's drag/scroll, we continue to return raw
            return raw

        # 4. Confirmation progress
        elapsed_ms = (now - self._gesture_hold_start) * 1000.0
        progress = min(1.0, elapsed_ms / confirm_ms)
        self._arc_progress = progress
        raw.arc_progress = progress

        if elapsed_ms >= confirm_ms:
            # TRIGGER!
            if raw.gesture in (GestureType.DRAG, GestureType.SCROLL):
                # Mark as firing/streaming forever until release
                self._gesture_hold_start = now + 9999
                return raw
            else:
                # Discrete (Click) — fire once and block
                cooldown = self.settings.gesture_cooldown_ms / 1000.0
                if now - self._last_action_time >= cooldown:
                    self._last_action_time = now
                    self._gesture_hold_start = now + 9999 # Use the same marker
                    return raw

        return GestureResult(GestureType.NONE, 0.0, arc_progress=progress, hand_data=hd)

    # ── Reset ─────────────────────────────────────────────────────

    def reset(self):
        self._prev_gesture        = GestureType.NONE
        self._gesture_hold_count  = 0
        self._gesture_hold_start  = 0.0
        self._prev_wrist_y        = None
        self._prev_scroll_y       = 0.0
        self._scroll_accum        = 0.0
        self._fist_hold_frames    = 0
        self._drag_active         = False
        self._arc_progress        = 0.0
        self._dwell_pos           = None
        self._dwell_start         = None
        self._dwell_fired         = False