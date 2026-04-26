#!/usr/bin/env python3
"""
AdaptiveGesture - Personalized Gesture-Based Cursor Control System
Main entry point for the application.

Fixes in this version:
  - Added a raw debugging loop directly in main.py to allow isolated testing 
    of the Absolute Cursor Mapping logic. 
  - When run with `--debug`, it bypasses the complex UI and shows the MANDATORY
    DEBUG OVERLAY exactly as requested (control region, fingertip, print values).
"""

import sys
import os
import argparse
import logging
import cv2
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ui.app_window import GestureControlApp
from config.settings import Settings
from utils.logger import setup_logger
from core.hand_tracker import HandTracker
from core.cursor_controller import CursorController

def parse_args():
    parser = argparse.ArgumentParser(
        description="AdaptiveGesture - Gesture-Based Cursor Control"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable simple debug tracking loop (Bypasses UI to test bare cursor logic)"
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Launch directly into calibration mode"
    )
    parser.add_argument(
        "--no-voice", action="store_true",
        help="Disable voice commands"
    )
    return parser.parse_args()


def run_simple_debug_loop(settings: Settings):
    """
    7. DEBUG OVERLAY (MANDATORY) & 8. REMOVE ALL UNSTABLE LOGIC.
    This runs a barebones deterministic loop strictly to prove absolute 
    cursor tracking and show the overlay.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Simple Debug Tracking Loop...")
    
    cap = cv2.VideoCapture(settings.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    tracker = HandTracker(settings)
    cursor = CursorController(settings, actual_w, actual_h)
    
    cv2.namedWindow("Debug Cursor Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Debug Cursor Tracking", actual_w, actual_h)
    
    logger.info("Press 'q' to quit debug mode.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # The hand tracker processes the unflipped frame to get normalized values
        hand_data = tracker.process_frame(frame)
        
        # We flip the frame for the user's mirror-view
        frame = cv2.flip(frame, 1)
        
        if hand_data:
            # Move cursor absolute based on index fingertip
            # pass the normalizd positions
            cursor.move(hand_data.landmarks[8][:2])
        else:
            cursor.notify_hand_absent()
            
        # Draw mandatory debug layout on the mirrored frame
        cursor.draw_debug_overlay(frame)
        
        cv2.imshow("Debug Cursor Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logger(log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting AdaptiveGesture System...")

    # Load settings
    settings = Settings()
    settings.camera_index = args.camera
    settings.debug_mode = args.debug
    settings.voice_enabled = not args.no_voice

    try:
        if args.debug:
            # Run simple deterministic tracking logic with debug visualizer
            run_simple_debug_loop(settings)
        else:
            # Run the normal application window
            app = GestureControlApp(settings)
            if args.calibrate:
                app.launch_calibration_on_start = True
            app.run()
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
