"""
Test script for Scroll and Drag functionality.
Tests:
  - Fist-to-Drag logic (mouseDown when fist closed, mouseUp otherwise)
  - Index-tip-movement scroll logic (scroll based on Y delta when 2+ fingers are up)
"""

import cv2
import sys
import os
import time
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from core.hand_tracker import HandTracker
from core.cursor_controller import CursorController
from core.gesture_recognizer import GestureRecognizer, GestureType
from core.action_executor import ActionExecutor
from config.settings import Settings
from calibration.gesture_store import GestureStore

def main():
    print("Initializing Scroll & Drag Test...")
    settings = Settings()
    # Ensure some defaults for testing
    settings.gesture_confirm_ms = 100 # Fast response for testing
    
    # Initialize Core Modules
    gesture_store = GestureStore()
    tracker = HandTracker(settings)
    recognizer = GestureRecognizer(settings, gesture_store)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Warm up camera to get dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        return
        
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cursor = CursorController(settings, actual_w, actual_h)
    executor = ActionExecutor(cursor)
    
    print(f"Webcam resolution: {actual_w}x{actual_h}")
    print("\nGESTURE CONTROLS:")
    print("  - FIST (Close hand)   -> DRAG")
    print("  - 2 FINGERS UP        -> SCROLL (Move hand up/down to scroll)")
    print("  - 1 FINGER UP         -> MOVE CURSOR")
    print("\nPress 'q' to quit.")

    cv2.namedWindow("Scroll & Drag Test", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame for intuitive display
        frame = cv2.flip(frame, 1)
        
        # 1. Track Hand
        hand_data = tracker.process_frame(frame)

        status_text = "No Hand"
        
        if hand_data:
            # Always update cursor position first (mandatory for absolute mapping)
            cursor.update(hand_data.landmarks_px[8])
            
            # 2. Recognize and Execute
            result = recognizer.recognize(hand_data)
            status_text = executor.execute(result)
            
            # RAW DEBUG TO CONSOLE
            if result.gesture != GestureType.NONE:
                print(f"[DEBUG] Gesture detected: {result.gesture.name} -> Status: {status_text}")
            
            if not status_text:
                status_text = "Idle"
                
            # Draw landmarks
            tracker.draw_landmarks(frame, hand_data)
        else:
            cursor.notify_hand_lost()
            recognizer.reset()

        # Debug Overlay
        cursor.draw_debug(frame, status_text)
        
        # Additional state indicators
        is_dragging = cursor.is_dragging
        drag_col = (0, 255, 0) if is_dragging else (0, 0, 255)
        cv2.putText(frame, f"MOUSE DOWN: {is_dragging}", (20, actual_h - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, drag_col, 2)

        cv2.imshow("Scroll & Drag Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Ensure mouse is released on exit
    cursor.stop_drag()
    
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("Test finished.")

if __name__ == "__main__":
    main()
