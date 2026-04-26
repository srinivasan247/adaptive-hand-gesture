"""
OverlayRenderer — Handles all visual feedback and HUD elements.
Features:
  - Glassmorphism-style status indicators
  - Dynamic gesture name and confidence display
  - Arc-based progress for time-based gestures (confirmation/dwell)
  - Control zone visualization
  - Voice assistant status and command flashing
"""

import cv2
import numpy as np
import time
import math
from typing import Optional, Tuple

class OverlayRenderer:
    def __init__(self, settings, width: int, height: int, ctrl_zone: Tuple[int, int, int, int]):
        self.settings = settings
        self.width = width
        self.height = height
        self.ctrl_zone = ctrl_zone
        
        # State for status messages
        self.status_text = ""
        self.status_color = (255, 255, 255)
        self.status_end_time = 0.0
        
        # State for voice command flashing
        self.voice_flash_text = ""
        self.voice_flash_end_time = 0.0
        
        # Colors (BGR) - Professional Dark Blue Theme
        self.CLR_ACCENT = (255, 170, 0)      # Deep Azure/Blue
        self.CLR_SUCCESS = (180, 255, 140)   # Soft Green
        self.CLR_WARNING = (140, 140, 255)   # Soft Red
        self.CLR_HUD = (50, 30, 20)          # Rich Navy/Dark Blue (BGR)
        self.CLR_TEXT = (250, 250, 250)      # White
        self.CLR_VOICE = (255, 220, 100)     # Light Cyan/Blue for voice
        self.CLR_BORDER = (100, 70, 40)      # Dark blue border

    def show_status(self, text: str, duration: float = 2.0, color: Optional[Tuple[int, int, int]] = None):
        """Display a status message on screen."""
        self.status_text = text
        self.status_end_time = time.time() + duration
        if color:
            self.status_color = color
        else:
            self.status_color = self.CLR_TEXT

    def flash_voice_command(self, phrase: str):
        """Show a brief flash of a recognized voice command."""
        self.voice_flash_text = f'"{phrase.upper()}"'
        self.voice_flash_end_time = time.time() + 1.2

    def render(self, frame, hand_data, gesture_result, calib_mgr, voice_active=False, is_paused=False):
        """Main render loop for the overlay."""
        overlay = frame.copy()
        now = time.time()

        # 1. Background Blur/Overlay for glass effect if needed
        # (Actually, just standard drawing is faster for OpenCV)

        # 2. Control Zone Bounds
        if self.settings.show_cursor_zone:
            x1, y1, x2, y2 = self.ctrl_zone
            # Mirror X-coords for display (frame is mirrored)
            disp_x1 = self.width - x2
            disp_x2 = self.width - x1
            cv2.rectangle(frame, (disp_x1, y1), (disp_x2, y2), (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "CONTROL REGION", (disp_x1 + 5, y1 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # 3. Status Message
        if now < self.status_end_time:
            self._draw_status_bar(frame, self.status_text, self.status_color)

        # 4. Voice Assistant HUD
        self._draw_voice_hud(frame, voice_active, now)

        # 5. Hand Landmarks & Gesture Info
        if hand_data and not is_paused:
            # We don't draw landmarks here if hand_tracker.draw_landmarks was called in main.py
            # But we can draw the fingertip focus
            idx_tip = hand_data.landmarks_px[8]
            disp_x = self.width - idx_tip[0]
            disp_y = idx_tip[1]
            
            # Draw tracking point
            cv2.circle(frame, (disp_x, disp_y), 6, self.CLR_ACCENT, -1, cv2.LINE_AA)
            cv2.circle(frame, (disp_x, disp_y), 10, self.CLR_ACCENT, 1, cv2.LINE_AA)

            if gesture_result:
                g_name = gesture_result.custom_name or gesture_result.gesture.name
                if g_name != "NONE":
                    # Draw Gesture Label near fingertip
                    cv2.putText(frame, g_name, (disp_x + 15, disp_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.CLR_ACCENT, 2, cv2.LINE_AA)
                    
                    # Draw Progress Arc if applicable
                    if hasattr(gesture_result, 'arc_progress') and gesture_result.arc_progress < 1.0:
                        self._draw_progress_arc(frame, (disp_x, disp_y), gesture_result.arc_progress)

        # 6. Pause State
        if is_paused:
            self._draw_centered_rect(frame, "PAUSED", self.CLR_WARNING)

        # 7. Calibration State
        if calib_mgr and calib_mgr.is_active:
            self._draw_calibration_hud(frame, calib_mgr)

        return frame

    def _draw_status_bar(self, frame, text, color):
        # Top bar style - Sleek semi-transparent with glow
        h, w = frame.shape[:2]
        # Draw background bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (20, 10, 5), -1) # Darker Navy
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Bottom accent line
        cv2.line(frame, (0, 40), (w, 40), color, 1, cv2.LINE_AA)
        
        # Center text with slight shadow for readability
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(frame, text, (w // 2 - tw // 2 + 1, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, text, (w // 2 - tw // 2, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def _draw_voice_hud(self, frame, active, now):
        h, w = frame.shape[:2]
        v_color = self.CLR_VOICE if active else (120, 120, 120)
        
        # Modern bottom-left pill
        pill_w, pill_h = 130, 34
        px, py = 25, h - 55
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pill_w, py + pill_h), (30, 15, 5), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pill_w, py + pill_h), v_color, 1, cv2.LINE_AA)
        
        # Pulse/Status Icon
        icon_cx, icon_cy = px + 20, py + pill_h // 2
        if active:
            pulse = (math.sin(now * 6) + 1) / 2
            cv2.circle(frame, (icon_cx, icon_cy), int(6 + 4 * pulse), v_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (icon_cx, icon_cy), int(12 + 2 * pulse), v_color, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (icon_cx, icon_cy), 6, v_color, 1, cv2.LINE_AA)
            
        cv2.putText(frame, "ASSISTANT", (px + 42, py + 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, v_color, 1, cv2.LINE_AA)

        # Voice Command Flash
        if now < self.voice_flash_end_time:
            tw = cv2.getTextSize(self.voice_flash_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
            # Dark backing
            cv2.rectangle(frame, (w // 2 - tw // 2 - 20, h - 100), (w // 2 + tw // 2 + 20, h - 60), (0, 0, 0), -1)
            cv2.putText(frame, self.voice_flash_text, (w // 2 - tw // 2, h - 73), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.CLR_VOICE, 2, cv2.LINE_AA)

    def _draw_progress_arc(self, frame, center, progress):
        """Draw an 360-degree arc showing progress."""
        radius = 25
        # Draw background circle
        cv2.circle(frame, center, radius, (80, 80, 80), 2, cv2.LINE_AA)
        # Draw arc
        angle = int(progress * 360)
        cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, self.CLR_ACCENT, 3, cv2.LINE_AA)

    def _draw_centered_rect(self, frame, text, color):
        h, w = frame.shape[:2]
        tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        tx, ty = w // 2 - tw // 2, h // 2 + th // 2
        
        # Semi-transparent box
        sub = frame[ty - th - 30 : ty + 20, tx - 30 : tx + tw + 30]
        black = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.4, black, 0.6, 0, sub)
        
        cv2.rectangle(frame, (tx - 30, ty - th - 30), (tx + tw + 30, ty + 20), color, 2, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

    def _draw_calibration_hud(self, frame, mgr):
        h, w = frame.shape[:2]
        # Darken entire frame slightly
        black = np.zeros_like(frame)
        cv2.addWeighted(frame, 0.7, black, 0.3, 0, frame)
        
        # Instruction text
        instr = f"CALIBRATING: {mgr.gesture_name.upper()}"
        tw = cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.putText(frame, instr, (w // 2 - tw // 2, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.CLR_ACCENT, 2, cv2.LINE_AA)
        
        if not mgr.is_collecting:
            sub = "Hold hand in position and press SPACE"
            tw2 = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            cv2.putText(frame, sub, (w // 2 - tw2 // 2, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            # Collection progress bar
            prog = mgr.get_progress()
            bw, bh = 300, 20
            bx, by = w // 2 - bw // 2, 150
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
            cv2.rectangle(frame, (bx, by), (bx + int(bw * prog), by + bh), self.CLR_SUCCESS, -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
            
            p_text = f"Capturing... {int(prog*100)}%"
            cv2.putText(frame, p_text, (bx, by - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
