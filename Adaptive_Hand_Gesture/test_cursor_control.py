"""
Independent verification module for cursor control.
Tests: Absolute Mapping, Smoothing, and Minimal Clicks.
Usage: python test_cursor_control.py
"""

import cv2
import sys
import os
import math
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from core.hand_tracker import HandTracker
from core.cursor_controller import CursorController

# Configuration
PINCH_THRESHOLD_PX = 40

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

class MockSettings:
    """Minimal settings object for HandTracker."""
    def __init__(self):
        self.max_hands = 1
        self.detection_confidence = 0.7
        self.tracking_confidence = 0.7
        self.pinch_threshold = 0.05 # For internal HandTracker use

def main():
    print("Initializing Clean Cursor Control Test...")
    settings = MockSettings()
    
    # Initialize Core Modules
    tracker = HandTracker(settings)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Warm up camera to get dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        return
        
    h, w = frame.shape[:2]
    cursor = CursorController(w, h)
    
    print(f"Webcam resolution: {w}x{h}")
    print("Starting test loop. Press 'q' to quit.")

    left_clicked = False
    right_clicked = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Track Hand
        hand_data = tracker.process_frame(frame)

        # Mirror frame for intuitive display
        display_frame = cv2.flip(frame, 1)
        
        gesture_name = "None"

        if hand_data:
            # PHASE 2: Gestures
            # Landmarks are: 4=Thumb Tip, 8=Index Tip, 12=Middle Tip
            # HandTracker provides landmarks_px (pixel coordinates)
            landmarks_px = hand_data.landmarks_px
            idx_tip = tuple(landmarks_px[8])
            thumb_tip = tuple(landmarks_px[4])
            mid_tip = tuple(landmarks_px[12])

            # MOVE (Index Finger)
            cursor.update(idx_tip)
            gesture_name = "Moving"

            # LEFT CLICK (Index + Thumb Pinch)
            dist_left = calculate_distance(idx_tip, thumb_tip)
            if dist_left < PINCH_THRESHOLD_PX:
                gesture_name = "Left Click (Pinch)"
                if not left_clicked:
                    cursor.click_left()
                    left_clicked = True
            else:
                left_clicked = False

            # RIGHT CLICK (Index + Middle Pinch)
            dist_right = calculate_distance(idx_tip, mid_tip)
            if dist_right < PINCH_THRESHOLD_PX and not left_clicked:
                gesture_name = "Right Click (Pinch)"
                if not right_clicked:
                    cursor.click_right()
                    right_clicked = True
            else:
                right_clicked = False
        else:
            cursor.notify_hand_lost()

        # PHASE 5: DEBUG VISUALIZATION
        cursor.draw_debug(display_frame, gesture_name)
        
        cv2.imshow("Cursor Control Test (Minimal)", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("Test finished.")

if __name__ == "__main__":
    main()
