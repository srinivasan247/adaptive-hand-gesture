"""
Main application window — orchestrates all subsystems.

New flow vs original:
  1. Open camera
  2. If onboarding not complete → run OnboardingWizard
  3. Apply finger_count → set gesture slots
  4. Main gesture control loop
"""

import cv2
import numpy as np
import logging
import time
import threading
import os
import tkinter as tk
from typing import Optional

from config.settings import Settings
from core.hand_tracker import HandTracker
from core.gesture_recognizer import GestureRecognizer, GestureType
from core.cursor_controller import CursorController
from core.action_executor import ActionExecutor
from calibration.gesture_store import GestureStore
from calibration.calibration_manager import CalibrationManager
from voice.voice_handler import VoiceCommandHandler
from ui.overlay_renderer import OverlayRenderer
from ui.onboarding import OnboardingWizard, FINGER_COUNT_SLOTS
from ui.calibration_dialog import CalibrationDialog
from ui.settings_dialog import SettingsDialog

logger = logging.getLogger(__name__)

WINDOW_NAME = "AdaptiveGesture Control"
KEY_QUIT     = ord("q")
KEY_PAUSE    = ord("p")
KEY_CALIBRATE = ord("c")
KEY_SETTINGS  = ord("s")
KEY_SPACE    = 32
KEY_ESC      = 27
KEY_RESET_SETUP = ord("o")   # 'o' for onboarding — re-run setup
KEY_VOICE    = ord("v")


class GestureControlApp:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.ensure_dirs()

        self.launch_calibration_on_start = False
        self._paused  = False
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None

        self.hand_tracker:        Optional[HandTracker]        = None
        self.gesture_recognizer:  Optional[GestureRecognizer]  = None
        self.cursor:              Optional[CursorController]   = None
        self.executor:            Optional[ActionExecutor]     = None
        self.overlay:             Optional[OverlayRenderer]    = None
        self.gesture_store:       Optional[GestureStore]       = None
        self.calib_mgr:           Optional[CalibrationManager] = None
        self.voice:               Optional[VoiceCommandHandler] = None

        logger.info("GestureControlApp created.")

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        logger.info("Initializing subsystems...")

        # ── Camera ────────────────────────────────────────────────────────
        self._cap = cv2.VideoCapture(self.settings.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.settings.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.settings.fps)

        if not self._cap.isOpened():
            logger.error(f"Cannot open camera {self.settings.camera_index}")
            return

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened: {actual_w}x{actual_h}")

        # ── Core subsystems ───────────────────────────────────────────────
        self.gesture_store      = GestureStore()
        self.hand_tracker       = HandTracker(self.settings, actual_w, actual_h)
        self.gesture_recognizer = GestureRecognizer(self.settings, self.gesture_store)
        self.cursor             = CursorController(self.settings, actual_w, actual_h)
        self.executor           = ActionExecutor(self.cursor)
        self.calib_mgr          = CalibrationManager(self.gesture_store)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, actual_w, actual_h)

        # ── Onboarding ────────────────────────────────────────────────────
        if not self.settings.onboarding_complete:
            wizard = OnboardingWizard(
                self._cap, self.hand_tracker,
                self.gesture_store, self.settings,
                window_name=WINDOW_NAME,
            )
            finger_count = wizard.run()

            if finger_count is None:
                logger.info("Onboarding skipped — using defaults.")
                finger_count = 5

            # Save profile
            self.settings.user_finger_count   = finger_count
            self.settings.active_slots        = list(
                FINGER_COUNT_SLOTS.get(finger_count, FINGER_COUNT_SLOTS[5])
            )
            self.settings.onboarding_complete = True
            self.settings.save()
            logger.info(f"Onboarding complete. fingers={finger_count}, "
                        f"slots={self.settings.active_slots}")
            
            # RE-INITIALIZE CursorController with new calibration bounds
            self.cursor = CursorController(self.settings, actual_w, actual_h)
            self.executor.cursor = self.cursor

        # Apply slots to recognizer
        self.gesture_recognizer.set_active_slots(self.settings.active_slots)

        # ── Overlay ───────────────────────────────────────────────────────
        self.overlay = OverlayRenderer(
            self.settings, actual_w, actual_h,
            self.cursor.get_control_zone(),
        )

        # ── Voice ─────────────────────────────────────────────────────────
        if self.settings.voice_enabled:
            self.voice = VoiceCommandHandler(self.settings, self._on_voice_command)
            if self.voice.is_available:
                self.voice.start()

        if self.executor:
            self.executor.register_custom_action("stop_gesture",  self._pause)
            self.executor.register_custom_action("pause_gesture", self._pause)
            self.executor.register_custom_action("resume_gesture", self._resume)
            self.executor.register_custom_action(
                "start_calibration", lambda: self._show_calibration_dialog()
            )

        self._running = True
        self._print_help()

        # ── Main loop ─────────────────────────────────────────────────────
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Flip frame early so both tracking and visualization use mirrored coordinates
            # This matches the successful test_cursor_control.py logic.
            frame = cv2.flip(frame, 1)

            hand_data      = None
            gesture_result = None

            if not self._paused:
                hand_data = self.hand_tracker.process_frame(frame)

                if hand_data is None:
                    # Hand absent: freeze cursor, clear transient state
                    self.cursor.notify_hand_absent()
                    self.gesture_recognizer._prev_wrist_y = None
                    self.gesture_recognizer._scroll_accum  = 0.0
                    if self.cursor.is_dragging:
                        self.cursor.stop_drag(force=True)
                        self.gesture_recognizer._drag_active = False
                else:
                    if self.settings.show_landmarks:
                        self.hand_tracker.draw_landmarks(frame, hand_data)

                    if self.calib_mgr.is_active:
                        self.calib_mgr.process_frame(hand_data)
                    else:
                        # Always update cursor position first (like the test script did)
                        # but only if we are in a gesture slot that allows movement
                        # Actually, to match test script exactly: just always update.
                        self.cursor.update(hand_data.landmarks_px[8])

                        gesture_result = self.gesture_recognizer.recognize(hand_data)
                        if gesture_result:
                            self.executor.execute(gesture_result)

            frame = self.overlay.render(
                frame, hand_data, gesture_result, self.calib_mgr,
                voice_active=(self.voice is not None and self.voice.is_listening),
                is_paused=self._paused,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)

        self._shutdown()

    # ── Key handling ──────────────────────────────────────────────────────────

    def _handle_key(self, key: int):
        if key in (KEY_QUIT, KEY_ESC):
            self._running = False

        elif key == KEY_PAUSE:
            self._resume() if self._paused else self._pause()

        elif key == KEY_CALIBRATE:
            if not self.calib_mgr.is_active:
                self._show_calibration_dialog()
            else:
                self.calib_mgr.cancel()
                self.overlay.show_status("Calibration cancelled", 2.0)

        elif key == KEY_SPACE:
            if self.calib_mgr.is_active:
                self.calib_mgr.confirm_ready()

        elif key == KEY_SETTINGS:
            self._show_settings_dialog()

        elif key == KEY_RESET_SETUP:
            self._reset_and_rerun_onboarding()

        elif key == ord("h"):
            self._print_help()

        elif key == ord("r"):
            self.gesture_recognizer.reset()
            self.cursor.reset_smoothing()
            self.overlay.show_status("State reset", 1.5)

        elif key == KEY_VOICE:
            if self.voice:
                self.voice.toggle_listening()
                if self.voice.is_listening:
                    self.overlay.show_status("Voice: LISTENING", 2.0, (50, 220, 50))
                else:
                    self.overlay.show_status("Voice: MUTED", 2.0, (50, 50, 220))

    # ── Pause / resume ────────────────────────────────────────────────────────

    def _pause(self):
        self._paused = True
        if self.cursor.is_dragging:
            self.cursor.stop_drag()
        logger.info("Paused.")

    def _resume(self):
        self._paused = False
        self.gesture_recognizer.reset()
        logger.info("Resumed.")

    # ── Re-run onboarding ─────────────────────────────────────────────────────

    def _reset_and_rerun_onboarding(self):
        """Clear profile and run wizard again without restarting the app."""
        self.settings.reset_profile()
        self.gesture_store.clear()

        # Refresh tracker size too
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.hand_tracker = HandTracker(self.settings, actual_w, actual_h)

        wizard = OnboardingWizard(
            self._cap, self.hand_tracker,
            self.gesture_store, self.settings,
            window_name=WINDOW_NAME,
        )
        finger_count = wizard.run()
        if finger_count is None:
            finger_count = 5

        self.settings.user_finger_count   = finger_count
        self.settings.active_slots        = list(
            FINGER_COUNT_SLOTS.get(finger_count, FINGER_COUNT_SLOTS[5])
        )
        self.settings.onboarding_complete = True
        self.settings.save()

        self.gesture_recognizer.set_active_slots(self.settings.active_slots)
        self.gesture_recognizer.reset()
        
        # Refresh components with new calibration
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cursor = CursorController(self.settings, actual_w, actual_h)
        self.executor.cursor = self.cursor
        self.overlay.ctrl_zone = self.cursor.get_control_zone()
        
        self.overlay.show_status("Setup complete! New calibration active.", 3.0)
        logger.info("Onboarding re-run complete.")

    # ── Voice command handler ─────────────────────────────────────────────────

    def _on_voice_command(self, phrase: str, action: str):
        """
        Central dispatcher for all voice commands.
        Runs on the voice background thread — all pyautogui calls are
        thread-safe, so no lock needed. Gesture control keeps running.
        """
        import pyautogui
        import platform
        import subprocess

        logger.info(f"Voice: '{phrase}' -> {action}")
        self.overlay.flash_voice_command(phrase)

        OS = platform.system()  # "Windows", "Darwin", "Linux"

        # ── Calibration / system control (need main-thread Tkinter) ──────────
        if action == "start_calibration":
            threading.Thread(target=self._show_calibration_dialog,
                             daemon=True).start()
            return
        if action == "reset_state":
            self.gesture_recognizer.reset()
            self.cursor.reset_smoothing()
            self.overlay.show_status("State reset", 1.5)
            return
        if action == "show_help":
            self._print_help()
            self.overlay.show_status("Help printed to console", 1.5)
            return

        # ── Clicks ────────────────────────────────────────────────────────────
        if action == "left_click":
            self.cursor.click_left()
        elif action == "right_click":
            self.cursor.click_right()
        elif action == "double_click":
            self.cursor.double_click()

        # ── Scroll ────────────────────────────────────────────────────────────
        elif action == "scroll_up":
            self.cursor.scroll("up", self.settings.scroll_speed)
        elif action == "scroll_down":
            self.cursor.scroll("down", self.settings.scroll_speed)
        elif action == "scroll_top":
            pyautogui.hotkey("ctrl", "Home")
        elif action == "scroll_bottom":
            pyautogui.hotkey("ctrl", "End")
        elif action == "page_up":
            pyautogui.press("pageup")
        elif action == "page_down":
            pyautogui.press("pagedown")

        # ── Drag ──────────────────────────────────────────────────────────────
        elif action == "drag_start":
            self.cursor.start_drag()
            self.overlay.show_status("Drag started — say 'drop' to release", 2.0)
        elif action == "drag_drop":
            self.cursor.stop_drag()
            self.overlay.show_status("Dropped", 1.0)

        # ── Cursor grid positioning ────────────────────────────────────────────
        # Jumps cursor to a screen zone — hand keeps controlling fine position
        elif action.startswith("cursor_"):
            sw, sh = pyautogui.size()
            zone_map = {
                "cursor_centre":        (sw // 2,      sh // 2),
                "cursor_top_left":      (sw // 6,      sh // 6),
                "cursor_top_right":     (sw * 5 // 6,  sh // 6),
                "cursor_bottom_left":   (sw // 6,      sh * 5 // 6),
                "cursor_bottom_right":  (sw * 5 // 6,  sh * 5 // 6),
                "cursor_top_centre":    (sw // 2,      sh // 6),
                "cursor_bottom_centre": (sw // 2,      sh * 5 // 6),
                "cursor_mid_left":      (sw // 6,      sh // 2),
                "cursor_mid_right":     (sw * 5 // 6,  sh // 2),
            }
            pos = zone_map.get(action)
            if pos:
                pyautogui.moveTo(pos[0], pos[1], _pause=False)
                # Sync cursor controller smooth state so gesture takes over smoothly
                self.cursor._smooth_x = float(pos[0])
                self.cursor._smooth_y = float(pos[1])
                self.cursor._last_moved_x = float(pos[0])
                self.cursor._last_moved_y = float(pos[1])
                self.cursor._hand_was_present = False  # force smooth sync on next move

        # ── Keyboard shortcuts ────────────────────────────────────────────────
        elif action == "key_copy":        pyautogui.hotkey("ctrl", "c")
        elif action == "key_paste":       pyautogui.hotkey("ctrl", "v")
        elif action == "key_cut":         pyautogui.hotkey("ctrl", "x")
        elif action == "key_undo":        pyautogui.hotkey("ctrl", "z")
        elif action == "key_redo":        pyautogui.hotkey("ctrl", "y")
        elif action == "key_select_all":  pyautogui.hotkey("ctrl", "a")
        elif action == "key_save":        pyautogui.hotkey("ctrl", "s")
        elif action == "key_find":        pyautogui.hotkey("ctrl", "f")
        elif action == "key_new_tab":     pyautogui.hotkey("ctrl", "t")
        elif action == "key_close_tab":   pyautogui.hotkey("ctrl", "w")
        elif action == "key_next_tab":    pyautogui.hotkey("ctrl", "tab")
        elif action == "key_prev_tab":    pyautogui.hotkey("ctrl", "shift", "tab")
        elif action == "key_new_window":  pyautogui.hotkey("ctrl", "n")
        elif action == "key_back":        pyautogui.hotkey("alt", "left")
        elif action == "key_forward":     pyautogui.hotkey("alt", "right")
        elif action == "key_refresh":     pyautogui.press("f5")
        elif action == "key_escape":      pyautogui.press("escape")
        elif action == "key_enter":       pyautogui.press("enter")
        elif action == "key_space":       pyautogui.press("space")
        elif action == "key_delete":      pyautogui.press("delete")
        elif action == "key_backspace":   pyautogui.press("backspace")
        elif action == "key_tab":         pyautogui.press("tab")
        elif action == "key_alt_tab":
            pyautogui.hotkey("alt", "tab")

        # ── Window management ─────────────────────────────────────────────────
        elif action == "minimize":
            self.executor._minimize_window()
        elif action == "maximize":
            self.executor._maximize_window()
        elif action == "restore_window":
            if OS == "Windows":   pyautogui.hotkey("win", "up")
            elif OS == "Darwin":  pass
            else:                 pyautogui.hotkey("super", "up")
        elif action == "close":
            self.executor._close_window()
        elif action == "show_desktop":
            if OS == "Windows":   pyautogui.hotkey("win", "d")
            elif OS == "Darwin":  pyautogui.hotkey("command", "mission_control")
            else:                 pyautogui.hotkey("super", "d")
        elif action == "task_manager":
            if OS == "Windows":   pyautogui.hotkey("ctrl", "shift", "escape")
            elif OS == "Darwin":  subprocess.Popen(["open", "-a", "Activity Monitor"])
            else:                 subprocess.Popen(["gnome-system-monitor"])

        # ── App launchers ─────────────────────────────────────────────────────
        elif action == "app_browser":
            if OS == "Windows":   subprocess.Popen("start chrome", shell=True)
            elif OS == "Darwin":  subprocess.Popen(["open", "-a", "Safari"])
            else:                 subprocess.Popen(["xdg-open", "https://google.com"])
        elif action == "app_files":
            if OS == "Windows":   subprocess.Popen("explorer", shell=True)
            elif OS == "Darwin":  subprocess.Popen(["open", os.path.expanduser("~")])
            else:                 subprocess.Popen(["xdg-open", os.path.expanduser("~")])
        elif action == "app_terminal":
            if OS == "Windows":   subprocess.Popen("cmd", shell=True)
            elif OS == "Darwin":  subprocess.Popen(["open", "-a", "Terminal"])
            else:                 subprocess.Popen(["x-terminal-emulator"])
        elif action == "app_calculator":
            if OS == "Windows":   subprocess.Popen("calc", shell=True)
            elif OS == "Darwin":  subprocess.Popen(["open", "-a", "Calculator"])
            else:                 subprocess.Popen(["gnome-calculator"])
        elif action == "app_settings":
            if OS == "Windows":   subprocess.Popen("control", shell=True)
            elif OS == "Darwin":  subprocess.Popen(["open", "-a", "System Preferences"])
            else:                 subprocess.Popen(["gnome-control-center"])

        # ── System ────────────────────────────────────────────────────────────
        elif action == "screenshot":
            self.executor._take_screenshot()
        elif action == "zoom_in":
            if OS == "Darwin":   pyautogui.hotkey("command", "=")
            else:                pyautogui.hotkey("ctrl", "=")
        elif action == "zoom_out":
            if OS == "Darwin":   pyautogui.hotkey("command", "-")
            else:                pyautogui.hotkey("ctrl", "-")
        elif action == "lock_screen":
            if OS == "Windows":  pyautogui.hotkey("win", "l")
            elif OS == "Darwin": subprocess.Popen(
                ["osascript", "-e",
                 'tell application "System Events" to keystroke "q" '
                 'using {command down, control down, option down}'])
            else:                subprocess.Popen(["loginctl", "lock-session"])
        elif action == "volume_up":
            if OS == "Windows":  pyautogui.press("volumeup")
            else:                pyautogui.press("audioraisemax")
        elif action == "volume_down":
            if OS == "Windows":  pyautogui.press("volumedown")
            else:                pyautogui.press("audiolowermax")
        elif action == "volume_mute":
            if OS == "Windows":  pyautogui.press("volumemute")
            else:                pyautogui.press("audiomute")

        # ── Gesture control flow ──────────────────────────────────────────────
        elif action == "pause_gesture":
            self._pause()
        elif action == "resume_gesture":
            self._resume()

        # ── Unknown / cmd: prefix ─────────────────────────────────────────────
        elif action.startswith("cmd:"):
            subprocess.Popen(action[4:].strip(), shell=True)

        # Show feedback for all handled actions
        if not action.startswith("cursor_"):
            self.overlay.show_status(f"Voice: {phrase}", 1.2)

    # ── Dialogs ───────────────────────────────────────────────────────────────

    def _show_calibration_dialog(self):
        def _open():
            root = tk.Tk()
            root.withdraw()
            dlg = CalibrationDialog(root, self.gesture_store,
                                    self._start_calibration)
            root.wait_window(dlg.root)
            root.destroy()
        threading.Thread(target=_open, daemon=True).start()

    def _start_calibration(self, name: str, action: str):
        def _complete(gesture):
            self.overlay.show_status(f"✓ Saved: {gesture.name}", 3.0,
                                     (50, 220, 50))
            # Add newly calibrated action to active slots
            if action not in self.settings.active_slots:
                self.settings.active_slots.append(action)
                self.gesture_recognizer.set_active_slots(
                    self.settings.active_slots
                )
                self.settings.save()

        def _cancel():
            self.overlay.show_status("Calibration cancelled", 2.0)

        self.calib_mgr.start(name, action, _complete, _cancel)

    def _show_settings_dialog(self):
        def _open():
            root = tk.Tk()
            root.withdraw()
            dlg = SettingsDialog(root, self.settings,
                                 self._on_settings_applied)
            root.wait_window(dlg.root)
            root.destroy()
        threading.Thread(target=_open, daemon=True).start()

    def _on_settings_applied(self, new_settings: Settings):
        self.settings = new_settings
        self.overlay.show_status("Settings applied!", 2.0)

    # ── Help ──────────────────────────────────────────────────────────────────

    def _print_help(self):
        slots = self.settings.active_slots
        fc    = self.settings.user_finger_count
        print("\n" + "="*55)
        print("  AdaptiveGesture — active profile")
        print("="*55)
        print(f"  Finger count : {fc}")
        print(f"  Active slots : {', '.join(slots)}")
        print("-"*55)
        print("  Keys:")
        print("  Q/ESC  Quit        P  Pause/Resume")
        print("  C      Calibrate   S  Settings")
        print("  O      Re-run setup wizard")
        print("  R      Reset state H  This help")
        print("  V      Toggle Voice Assistant")
        print("  SPACE  Confirm calibration gesture")
        print("="*55 + "\n")

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _shutdown(self):
        logger.info("Shutting down...")
        if self.voice:
            self.voice.stop()
        if self.cursor and self.cursor.is_dragging:
            self.cursor.stop_drag()
        if self.hand_tracker:
            self.hand_tracker.close()
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")