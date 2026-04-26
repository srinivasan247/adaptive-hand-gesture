"""
Voice command integration — runs on a background thread 24/7.
Gesture control and voice work simultaneously at all times.
Voice fills every action gap — cursor positioning, clicks, scrolls,
drag, typing shortcuts, app control, system actions.
"""

import logging
import threading
import time
from typing import Callable, Optional, Dict

logger = logging.getLogger(__name__)


# ── Complete command map ───────────────────────────────────────────────────────
# Format: "phrase the user says" -> "action_key"
# Action keys are handled in app_window._on_voice_command()

VOICE_COMMANDS: Dict[str, str] = {

    # ── Clicks ────────────────────────────────────────────────────────────────
    "click":                "left_click",
    "left click":           "left_click",
    "select":               "left_click",
    "tap":                  "left_click",
    "right click":          "right_click",
    "context menu":         "right_click",
    "menu":                 "right_click",
    "double click":         "double_click",
    "double tap":           "double_click",
    "open":                 "double_click",

    # ── Scroll ────────────────────────────────────────────────────────────────
    "scroll up":            "scroll_up",
    "up":                   "scroll_up",
    "go up":                "scroll_up",
    "scroll down":          "scroll_down",
    "down":                 "scroll_down",
    "go down":              "scroll_down",
    "scroll top":           "scroll_top",
    "top":                  "scroll_top",
    "go to top":            "scroll_top",
    "scroll bottom":        "scroll_bottom",
    "bottom":               "scroll_bottom",
    "go to bottom":         "scroll_bottom",
    "page up":              "page_up",
    "page down":            "page_down",

    # ── Drag ──────────────────────────────────────────────────────────────────
    "start drag":           "drag_start",
    "drag":                 "drag_start",
    "hold":                 "drag_start",
    "drop":                 "drag_drop",
    "release":              "drag_drop",
    "stop drag":            "drag_drop",

    # ── Cursor grid positioning (jump to screen zones) ────────────────────────
    # User says a zone name — cursor jumps there immediately.
    # Useful when hand is moving and they want a quick jump.
    "go centre":            "cursor_centre",
    "go center":            "cursor_centre",
    "centre":               "cursor_centre",
    "center":               "cursor_centre",
    "go top left":          "cursor_top_left",
    "top left":             "cursor_top_left",
    "go top right":         "cursor_top_right",
    "top right":            "cursor_top_right",
    "go bottom left":       "cursor_bottom_left",
    "bottom left":          "cursor_bottom_left",
    "go bottom right":      "cursor_bottom_right",
    "bottom right":         "cursor_bottom_right",
    "go top":               "cursor_top_centre",
    "go bottom":            "cursor_bottom_centre",
    "go left":              "cursor_mid_left",
    "go right":             "cursor_mid_right",

    # ── Typing / clipboard ────────────────────────────────────────────────────
    "copy":                 "key_copy",
    "paste":                "key_paste",
    "cut":                  "key_cut",
    "undo":                 "key_undo",
    "redo":                 "key_redo",
    "select all":           "key_select_all",
    "save":                 "key_save",
    "find":                 "key_find",
    "new tab":              "key_new_tab",
    "close tab":            "key_close_tab",
    "next tab":             "key_next_tab",
    "previous tab":         "key_prev_tab",
    "go back":              "key_back",
    "go forward":           "key_forward",
    "refresh":              "key_refresh",
    "escape":               "key_escape",
    "enter":                "key_enter",
    "space bar":            "key_space",
    "delete":               "key_delete",
    "backspace":            "key_backspace",
    "tab":                  "key_tab",

    # ── Window management ─────────────────────────────────────────────────────
    "minimize":             "minimize",
    "minimise":             "minimize",
    "maximize":             "maximize",
    "maximise":             "maximize",
    "restore":              "restore_window",
    "close window":         "close",
    "close app":            "close",
    "close":                "close",
    "switch window":        "key_alt_tab",
    "alt tab":              "key_alt_tab",
    "next window":          "key_alt_tab",
    "show desktop":         "show_desktop",
    "task manager":         "task_manager",
    "new window":           "key_new_window",

    # ── App launchers ─────────────────────────────────────────────────────────
    "open browser":         "app_browser",
    "browser":              "app_browser",
    "open file manager":    "app_files",
    "file manager":         "app_files",
    "open terminal":        "app_terminal",
    "terminal":             "app_terminal",
    "open calculator":      "app_calculator",
    "calculator":           "app_calculator",
    "open settings":        "app_settings",

    # ── System ────────────────────────────────────────────────────────────────
    "screenshot":           "screenshot",
    "take screenshot":      "screenshot",
    "screen shot":          "screenshot",
    "zoom in":              "zoom_in",
    "zoom out":             "zoom_out",
    "lock screen":          "lock_screen",
    "volume up":            "volume_up",
    "volume down":          "volume_down",
    "mute":                 "volume_mute",
    "unmute":               "volume_mute",

    # ── Gesture control flow ──────────────────────────────────────────────────
    "pause":                "pause_gesture",
    "stop":                 "pause_gesture",
    "resume":               "resume_gesture",
    "continue":             "resume_gesture",
    "start calibration":    "start_calibration",
    "calibrate":            "start_calibration",
    "reset":                "reset_state",
    "help":                 "show_help",
}


class VoiceCommandHandler:
    """
    Background-thread voice listener.
    Runs continuously alongside gesture control — both work at the same time.
    """

    def __init__(self, settings, on_command: Callable[[str, str], None]):
        self.settings   = settings
        self.on_command = on_command

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active = False
        self._is_listening = True # Starts listening if enabled
        self._last_command      = ""
        self._last_command_time = 0.0

        self._sr_available = False
        try:
            import speech_recognition as sr
            self._sr         = sr
            self._recognizer = sr.Recognizer()
            # Lower energy threshold for quieter environments
            self._recognizer.energy_threshold        = 250
            self._recognizer.dynamic_energy_threshold = True
            # Short pause threshold — responds quickly
            self._recognizer.pause_threshold         = 0.4
            self._recognizer.non_speaking_duration   = 0.3
            self._sr_available = True
            logger.info("SpeechRecognition available.")
        except ImportError:
            logger.warning("SpeechRecognition not installed — voice disabled. "
                           "pip install SpeechRecognition pyaudio")

    @property
    def is_available(self) -> bool:
        return self._sr_available

    @property
    def is_running(self) -> bool:
        return self._active

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    def toggle_listening(self):
        self._is_listening = not self._is_listening
        if self._is_listening:
            logger.info("Voice listener unmuted.")
        else:
            logger.info("Voice listener muted.")

    def start(self):
        if not self._sr_available or self._active:
            return
        self._stop_event.clear()
        self._active = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("Voice listener started (running alongside gesture control).")

    def stop(self):
        if not self._active:
            return
        self._stop_event.set()
        self._active = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("Voice listener stopped.")

    def _listen_loop(self):
        sr = self._sr
        try:
            with sr.Microphone() as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=1.0)
                logger.info("Voice: mic calibrated, always listening...")

                while not self._stop_event.is_set():
                    if not self._is_listening:
                        time.sleep(0.1)
                        continue

                    try:
                        audio = self._recognizer.listen(
                            source,
                            timeout=self.settings.voice_timeout,
                            phrase_time_limit=self.settings.voice_phrase_limit,
                        )
                        phrase = self._recognizer.recognize_google(
                            audio,
                            language=self.settings.voice_language,
                        ).lower().strip()

                        logger.debug(f"Voice heard: '{phrase}'")
                        self._dispatch(phrase)

                    except sr.WaitTimeoutError:
                        pass
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        logger.error(f"Voice API error: {e}")
                        time.sleep(2.0)
                    except Exception as e:
                        logger.error(f"Voice error: {e}")
                        time.sleep(1.0)

        except Exception as e:
            logger.error(f"Microphone error: {e}")
            self._active = False

    def _dispatch(self, phrase: str):
        now = time.time()
        # Debounce — ignore same command within 1 second
        if phrase == self._last_command and now - self._last_command_time < 1.0:
            return

        # Exact match first, then substring match
        action = VOICE_COMMANDS.get(phrase)
        if not action:
            for cmd, act in VOICE_COMMANDS.items():
                if cmd in phrase:
                    action = act
                    phrase = cmd
                    break

        if action:
            self._last_command      = phrase
            self._last_command_time = now
            logger.info(f"Voice → '{phrase}' : {action}")
            try:
                self.on_command(phrase, action)
            except Exception as e:
                logger.error(f"Voice callback error: {e}")
        else:
            logger.debug(f"No match for: '{phrase}'")

    def get_available_commands(self) -> Dict[str, str]:
        return dict(VOICE_COMMANDS)