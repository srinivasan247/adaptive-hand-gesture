"""
Demo mode - showcases gesture recognition without a real camera.
Uses synthetic hand landmark data to simulate gestures.
Useful for testing the pipeline on machines without a webcam.

Run with: python demo_mode.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import time
import logging

from config.settings import Settings
from core.hand_tracker import HandData
from core.gesture_recognizer import GestureRecognizer, GestureType
from core.cursor_controller import CursorController
from core.action_executor import ActionExecutor
from calibration.gesture_store import GestureStore
from utils.logger import setup_logger

setup_logger(logging.INFO)
logger = logging.getLogger(__name__)

FRAME_W, FRAME_H = 640, 480
WINDOW = "AdaptiveGesture DEMO"


def make_synthetic_hand(finger_states, frame_w=FRAME_W, frame_h=FRAME_H, pinch=False):
    """Generate synthetic HandData for demo purposes."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.8, 0.0]   # wrist

    # MCP positions
    mcp_x = [0.43, 0.47, 0.51, 0.55, 0.59]
    for i in range(5):
        base_y = 0.60
        x = mcp_x[i]
        lm[1 + i*4] = [x, base_y + 0.05, 0.0]
        lm[2 + i*4] = [x, base_y + 0.05, 0.0]
        lm[3 + i*4] = [x, base_y + 0.08, 0.0]

        if i == 0:  # Thumb
            lm[4][0] = lm[3][0] - (0.05 if finger_states[0] else -0.05)
            lm[4][1] = lm[3][1]
        else:
            if finger_states[i]:
                lm[i*4 + 4][1] = lm[i*4 + 2][1] - 0.08  # tip above pip = extended
            else:
                lm[i*4 + 4][1] = lm[i*4 + 2][1] + 0.05  # tip below = bent

    lm_px = (lm[:, :2] * np.array([frame_w, frame_h])).astype(np.int32)
    pinch_dist = 0.03 if pinch else 0.18

    return HandData(
        landmarks=lm,
        landmarks_px=lm_px,
        handedness="Right",
        fingers_up=list(finger_states),
        finger_count=sum(finger_states),
        index_tip=tuple(lm_px[8]),
        thumb_tip=tuple(lm_px[4]),
        wrist=tuple(lm_px[0]),
        pinch_distance=pinch_dist,
        is_pinching=pinch,
        hand_bbox=(100, 100, 300, 250),
        raw_landmarks=None,
    )


def draw_demo_frame(frame, gesture_name, finger_states, step, total_steps):
    """Draw a demo visualization frame."""
    h, w = frame.shape[:2]
    frame[:] = (25, 25, 35)  # Dark background

    # Title
    cv2.putText(frame, "AdaptiveGesture - DEMO MODE", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)

    # Gesture being demonstrated
    cv2.putText(frame, f"Gesture: {gesture_name}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 100), 2)

    # Finger indicators
    labels = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    colors_on = (50, 220, 50)
    colors_off = (80, 80, 80)
    for i, (state, label) in enumerate(zip(finger_states, labels)):
        color = colors_on if state else colors_off
        cx = 60 + i * 110
        cy = 180
        cv2.circle(frame, (cx, cy), 30, color, -1)
        cv2.putText(frame, "UP" if state else "DN",
                    (cx - 15, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        cv2.putText(frame, label[:3], (cx - 15, cy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Progress bar
    bar_x, bar_y = 20, h - 60
    bar_w = w - 40
    progress = step / total_steps
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * progress), bar_y + 20),
                  (100, 200, 100), -1)
    cv2.putText(frame, f"Demo step {step}/{total_steps}  (Press Q to quit)",
                (bar_x, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame


DEMO_SEQUENCE = [
    # (label, finger_states, hold_seconds, pinch)
    ("MOVE CURSOR — 1 finger",         [False, True,  False, False, False], 2.5, False),
    ("RIGHT CLICK — 2 fingers",        [False, True,  True,  False, False], 2.0, False),
    ("SCROLL — 3 fingers",             [False, True,  True,  True,  False], 2.0, False),
    ("LEFT CLICK — 4 fingers",         [False, True,  True,  True,  True],  2.0, False),
    ("OPEN PALM — all 5",              [True,  True,  True,  True,  True],  2.0, False),
    ("PINCH CLICK",                    [True,  True,  False, False, False], 2.0, True),
    ("DRAG (fist)",                    [False, False, False, False, False], 2.5, False),
    ("MOVE CURSOR — 1 finger",         [False, True,  False, False, False], 2.0, False),
]


def run_demo():
    settings = Settings()
    settings.gesture_confirm_frames = 1
    settings.gesture_cooldown_ms = 0
    settings.movement_threshold = 0.0

    gesture_store = GestureStore()
    recognizer = GestureRecognizer(settings, gesture_store)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, FRAME_W, FRAME_H)

    logger.info("Demo mode started. Press Q to quit.")

    total = len(DEMO_SEQUENCE)
    for step, (label, finger_states, hold_secs, pinch) in enumerate(DEMO_SEQUENCE, 1):
        start = time.time()
        while time.time() - start < hold_secs:
            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            hd = make_synthetic_hand(finger_states, pinch=pinch)

            # Run through recognizer
            result = recognizer.recognize(hd)
            recognized = result.gesture.name if result.gesture.name != "NONE" else "—"

            draw_demo_frame(frame, label, finger_states, step, total)

            # Show recognized gesture
            cv2.putText(frame, f"Recognized: {recognized}",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 50), 2)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(33) & 0xFF
            if key == ord("q") or key == 27:
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    logger.info("Demo complete.")


if __name__ == "__main__":
    run_demo()
