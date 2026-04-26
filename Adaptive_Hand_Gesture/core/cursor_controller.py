"""
Stable, minimal cursor controller for absolute screen mapping.
Refactored to meet strict minimal requirements:
- Absolute mapping (no delta)
- Control region with margins
- Simple exponential smoothing
"""

import pyautogui
import logging
import cv2
from typing import Tuple

logger = logging.getLogger(__name__)

# PyAutoGUI safety and performance
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

class CursorController:
    """
    Handles deterministic cursor movement and click execution.
    """

    def __init__(self, settings, frame_w: int, frame_h: int):
        self.settings = settings
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.screen_w, self.screen_h = pyautogui.size()

        # PHASE 3: CONTROL REGION
        if hasattr(settings, "calibration_bounds") and settings.calibration_bounds:
            self.ctrl_x1, self.ctrl_y1, self.ctrl_x2, self.ctrl_y2 = settings.calibration_bounds
            logger.info(f"Loaded Calibrated Bounds: {settings.calibration_bounds}")
        else:
            # Default to 25% edges as fallback
            margin_x = int(frame_w * 0.25)
            margin_y = int(frame_h * 0.25)
            self.ctrl_x1 = margin_x
            self.ctrl_y1 = margin_y
            self.ctrl_x2 = frame_w - margin_x
            self.ctrl_y2 = frame_h - margin_y
            logger.info(f"Using default 25% margin bounds: {self.ctrl_x1}, {self.ctrl_y1} ...")

        # PHASE 4: SIMPLE STABILITY
        self.alpha = 0.75  # Exponential smoothing factor (0.7-0.85)
        
        # State
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.first_frame = True
        self._is_dragging = False
        self._prev_scroll_y = 0.0
        self._drag_grace_frames = 0
        self.DRAG_GRACE_THRESHOLD = 5 # frames to wait before releasing mouseUp

        # Debug variables
        self.last_norm = (0.0, 0.0)
        self.last_pixel = (0, 0)

        logger.info(f"Initialized CursorController: region={self.ctrl_x1},{self.ctrl_y1} to {self.ctrl_x2},{self.ctrl_y2}")

    def update(self, index_tip_px: Tuple[int, int]):
        """
        Process raw finger pixel position and move cursor.
        """
        px, py = index_tip_px
        self.last_pixel = (px, py)

        # PHASE 3: Mapping logic
        # 1. Clamp to control region
        clamped_x = max(self.ctrl_x1, min(px, self.ctrl_x2))
        clamped_y = max(self.ctrl_y1, min(py, self.ctrl_y2))

        # 2. Normalize (0.0 to 1.0)
        norm_x = (clamped_x - self.ctrl_x1) / (self.ctrl_x2 - self.ctrl_x1)
        norm_y = (clamped_y - self.ctrl_y1) / (self.ctrl_y2 - self.ctrl_y1)
        
        self.last_norm = (norm_x, norm_y)

        # 3. Map to screen
        target_x = norm_x * self.screen_w
        target_y = norm_y * self.screen_h

        # PHASE 4: Smoothing
        if self.first_frame:
            self.smooth_x = target_x
            self.smooth_y = target_y
            self.first_frame = False
        else:
            self.smooth_x = (self.alpha * self.smooth_x) + ((1 - self.alpha) * target_x)
            self.smooth_y = (self.alpha * self.smooth_y) + ((1 - self.alpha) * target_y)

        # Execute movement
        try:
            pyautogui.moveTo(self.smooth_x, self.smooth_y)
        except Exception as e:
            logger.warning(f"Cursor move failed: {e}")

    def click_left(self):
        pyautogui.click(button='left')
        print("[ACTION] Left Click")

    def click_right(self):
        pyautogui.click(button='right')
        print("[ACTION] Right Click")


    def double_click(self):
        pyautogui.click(clicks=2)
        print("[ACTION] Double Click")

    def scroll(self, direction: str, amount: int = 50):
        """Scroll execution based on direction."""
        pyautogui.scroll(amount if direction == "up" else -amount)

    def scroll_with_movement(self, index_tip_y: int):
        """USER FIX: direction-based scroll using index tip Y diff."""
        if self._prev_scroll_y != 0:
            diff = index_tip_y - self._prev_scroll_y
            if abs(diff) > 5: # threshold
                self._prev_scroll_y = float(index_tip_y)
                if diff > 0: pyautogui.scroll(-50) # Scroll DOWN
                else:        pyautogui.scroll(50)  # Scroll UP
                return True
        self._prev_scroll_y = float(index_tip_y)
        return False

    def reset_scroll(self):
        self._prev_scroll_y = 0.0

    def start_drag(self):
        """Start drag action using mouseDown."""
        if not self._is_dragging:
            pyautogui.mouseDown()
            self._is_dragging = True
            print("[ACTION] Drag Started")

    def stop_drag(self):
        """Stop drag action using mouseUp."""
        if self._is_dragging:
            pyautogui.mouseUp()
            self._is_dragging = False
            print("[ACTION] Drag Released")

    @property
    def is_dragging(self):
        return self._is_dragging

    def notify_hand_lost(self):
        self.first_frame = True
        self.stop_drag()

    def notify_hand_absent(self):
        self.notify_hand_lost()

    def get_control_zone(self):
        return (self.ctrl_x1, self.ctrl_y1, self.ctrl_x2, self.ctrl_y2)

    def draw_debug(self, frame, gesture_name: str):
        """
        PHASE 5: DEBUG VISUALIZATION
        Draws control box, tracked point, and status info.
        Assumes frame is already mirrored for display.
        """
        # Mirror coordinates for drawing if frame is mirrored
        # In a mirrored frame, raw X becomes frame_w - X
        disp_x = self.frame_w - self.last_pixel[0]
        disp_y = self.last_pixel[1]
        
        box_x1 = self.frame_w - self.ctrl_x2
        box_x2 = self.frame_w - self.ctrl_x1

        # Control Region Box
        cv2.rectangle(frame, (box_x1, self.ctrl_y1), (box_x2, self.ctrl_y2), (255, 0, 255), 2)
        cv2.putText(frame, "Control Area", (box_x1, self.ctrl_y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Fingertip point
        cv2.circle(frame, (disp_x, disp_y), 8, (0, 255, 0), -1)
        cv2.circle(frame, (disp_x, disp_y), 12, (0, 0, 255), 2)

        # Status Overlay
        cv2.putText(frame, f"Gesture: {gesture_name}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Cursor: {int(self.smooth_x)}, {int(self.smooth_y)}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Norm Pos: {self.last_norm[0]:.2f}, {self.last_norm[1]:.2f}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
