"""
Action executor — translates GestureResults into system actions.
Updated to handle: dwell-click, wrist-move cursor, voice-mapped actions.
"""
 
import logging
import time
import subprocess
import platform
from typing import Optional
import pyautogui
 
from core.gesture_recognizer import GestureResult, GestureType
from core.cursor_controller import CursorController
 
logger = logging.getLogger(__name__)
OS = platform.system()
 
 
class ActionExecutor:
    def __init__(self, cursor: CursorController):
        self.cursor = cursor
        self._custom_actions = {
            "left_click":   self.cursor.click_left,
            "right_click":  self.cursor.click_right,
            "double_click": self.cursor.double_click,
            "scroll_up":    lambda: self.cursor.scroll("up"),
            "scroll_down":  lambda: self.cursor.scroll("down"),
            "screenshot":   self._take_screenshot,
            "minimize":     self._minimize_window,
            "maximize":     self._maximize_window,
            "close":        self._close_window,
        }
 
    def execute(self, result: GestureResult) -> str:
        g = result.gesture
        hd = result.hand_data
 
        # USER FIX: if action == "DRAG": start_drag() else: stop_drag()
        if g == GestureType.DRAG and hd:
            self.cursor.start_drag()
        else:
            self.cursor.stop_drag()

        # USER FIX: if action == "SCROLL": movement_logic else: reset_scroll()
        if g == GestureType.SCROLL and hd:
            self.cursor.scroll_with_movement(hd.landmarks_px[8][1])
            return "Scrolling"
        else:
            self.cursor.reset_scroll()

        if g == GestureType.NONE:
            return ""
 
        if g == GestureType.MOVE_CURSOR and hd:
            return "Moving cursor"
 
        if g == GestureType.WRIST_MOVE and hd:
            self._execute_wrist_move(result)
            return "Wrist cursor"
 
        if g == GestureType.LEFT_CLICK and hd:
            if result.custom_action == "double_click":
                self.cursor.double_click()
                return "Double click"
            self.cursor.click_left()
            return "Left click"
 
        if g == GestureType.RIGHT_CLICK and hd:
            self.cursor.click_right()
            return "Right click"

 
        if g == GestureType.DWELL_CLICK and hd:
            self.cursor.click_left()
            return "Dwell click"
 
        if g == GestureType.DRAG and hd:
            return "Dragging"
 
        if g == GestureType.SCROLL_UP:
            self.cursor.scroll("up")
            return "Scroll up"
 
        if g == GestureType.SCROLL_DOWN:
            self.cursor.scroll("down")
            return "Scroll down"
 
        if g == GestureType.OPEN_PALM:
            return "Open palm"
 
        if g == GestureType.CUSTOM:
            # If a custom gesture mapped to "scroll" is active
            if result.custom_action == "scroll" and hd:
                self.cursor.scroll_with_movement(hd.landmarks_px[8][1])
                return "Custom Scrolling"
            return self._execute_custom(result)
 
        return ""
 
    def _execute_wrist_move(self, result: GestureResult):
        """Move cursor based on wrist angle encoded in custom_name."""
        try:
            ax, ay = map(float, result.custom_name.split(","))
        except Exception:
            return
        # Scale angle to pixel delta
        cx, cy = pyautogui.position()
        speed = 400
        nx = max(0, min(pyautogui.size()[0] - 1, cx + int(ax * speed)))
        ny = max(0, min(pyautogui.size()[1] - 1, cy + int(ay * speed)))
        pyautogui.moveTo(nx, ny, _pause=False)
 
    def _execute_custom(self, result: GestureResult) -> str:
        action = result.custom_action or ""
        name = result.custom_name or "custom"
        if action in self._custom_actions:
            self._custom_actions[action]()
            return f"Custom [{name}]: {action}"
        if action.startswith("cmd:"):
            cmd = action[4:].strip()
            try:
                subprocess.Popen(cmd, shell=True)
                return f"Launched: {cmd}"
            except Exception as e:
                logger.error(f"Custom cmd failed: {e}")
        return f"Unknown action: {action}"
 
    def register_custom_action(self, name: str, handler):
        self._custom_actions[name] = handler
 
    def _take_screenshot(self):
        path = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        pyautogui.screenshot(path)
        logger.info(f"Screenshot: {path}")
 
    def _minimize_window(self):
        if OS == "Windows":
            pyautogui.hotkey("win", "down")
        elif OS == "Darwin":
            pyautogui.hotkey("command", "m")
        else:
            pyautogui.hotkey("super", "h")
 
    def _maximize_window(self):
        if OS == "Windows":
            pyautogui.hotkey("win", "up")
        elif OS == "Darwin":
            pass
        else:
            pyautogui.hotkey("super", "up")
 
    def _close_window(self):
        if OS == "Darwin":
            pyautogui.hotkey("command", "w")
        else:
            pyautogui.hotkey("alt", "F4")