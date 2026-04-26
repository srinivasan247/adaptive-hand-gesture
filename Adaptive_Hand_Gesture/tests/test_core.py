"""
Unit tests for core gesture recognition and calibration logic.
Run with: python -m pytest tests/ -v
No camera required.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_landmarks(finger_states=None):
    """
    Create a synthetic 21-landmark array.
    finger_states: list of 5 bools [thumb, index, middle, ring, pinky]
    """
    # Base hand pose (all fingers bent)
    lm = np.zeros((21, 3), dtype=np.float32)

    # Wrist
    lm[0] = [0.5, 0.8, 0.0]

    # Knuckles (MCP joints) - base positions
    mcps = {5: [0.45, 0.6], 9: [0.5, 0.58], 13: [0.55, 0.6], 17: [0.6, 0.63]}
    for idx, pos in mcps.items():
        lm[idx] = [pos[0], pos[1], 0.0]
        lm[idx+1] = [pos[0], pos[1]+0.05, 0.0]   # PIP
        lm[idx+2] = [pos[0], pos[1]+0.08, 0.0]   # DIP
        lm[idx+3] = [pos[0], pos[1]+0.10, 0.0]   # TIP (bent = below PIP)

    # Thumb
    lm[1] = [0.42, 0.68, 0.0]
    lm[2] = [0.40, 0.63, 0.0]
    lm[3] = [0.38, 0.60, 0.0]
    lm[4] = [0.37, 0.58, 0.0]  # thumb tip left of IP = extended (right hand)

    if finger_states:
        # thumb, index, middle, ring, pinky
        finger_map = [
            (4, 3),   # thumb tip, ip
            (8, 6),   # index tip, pip
            (12, 10), # middle
            (16, 14), # ring
            (20, 18), # pinky
        ]
        for i, (tip_idx, pip_idx) in enumerate(finger_map):
            if i == 0:  # thumb - use x axis
                if finger_states[0]:
                    lm[4][0] = lm[3][0] - 0.05  # tip left of IP
                else:
                    lm[4][0] = lm[3][0] + 0.05  # tip right of IP
            else:
                if finger_states[i]:
                    lm[tip_idx][1] = lm[pip_idx][1] - 0.05  # tip above PIP = extended
                else:
                    lm[tip_idx][1] = lm[pip_idx][1] + 0.05  # tip below PIP = bent

    return lm


def make_hand_data(finger_states=None, finger_count=None):
    """Create a mock HandData object."""
    from core.hand_tracker import HandData
    lm = make_landmarks(finger_states)
    lm_px = (lm[:, :2] * np.array([640, 480])).astype(np.int32)

    if finger_states is None:
        finger_states = [False, True, False, False, False]
    if finger_count is None:
        finger_count = sum(finger_states)

    return HandData(
        landmarks=lm,
        landmarks_px=lm_px,
        handedness="Right",
        fingers_up=finger_states,
        finger_count=finger_count,
        index_tip=tuple(lm_px[8]),
        thumb_tip=tuple(lm_px[4]),
        wrist=tuple(lm_px[0]),
        pinch_distance=0.15,
        is_pinching=False,
        hand_bbox=(100, 100, 200, 200),
        raw_landmarks=None,
    )


# ── Tests: GestureStore ───────────────────────────────────────────────────────

class TestGestureStore:

    def setup_method(self):
        """Use a temp directory for each test."""
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self._tmpdir.name)

        # Patch GESTURES_FILE
        import config.settings as cs
        self._orig = cs.GESTURES_FILE
        cs.GESTURES_FILE = tmp / "gestures.json"

        from calibration.gesture_store import GestureStore
        self.store = GestureStore()

    def teardown_method(self):
        import config.settings as cs
        cs.GESTURES_FILE = self._orig
        self._tmpdir.cleanup()

    def test_add_and_retrieve(self):
        from calibration.gesture_store import CustomGesture
        lm = make_landmarks()
        g = CustomGesture(name="Test", action="left_click", template=lm.tolist())
        self.store.add(g)
        assert len(self.store) == 1
        assert self.store.get("Test") is not None

    def test_remove(self):
        from calibration.gesture_store import CustomGesture
        lm = make_landmarks()
        g = CustomGesture(name="Del", action="left_click", template=lm.tolist())
        self.store.add(g)
        assert self.store.remove("Del") is True
        assert len(self.store) == 0

    def test_persistence(self):
        from calibration.gesture_store import CustomGesture, GestureStore
        import config.settings as cs
        lm = make_landmarks()
        g = CustomGesture(name="Persist", action="right_click", template=lm.tolist())
        self.store.add(g)

        # Load fresh
        store2 = GestureStore()
        assert len(store2) == 1
        assert store2.get("Persist").action == "right_click"

    def test_match_identical(self):
        from calibration.gesture_store import CustomGesture
        lm = make_landmarks()
        g = CustomGesture(name="Self", action="click", template=lm.tolist())
        score = g.match(lm)
        assert score > 0.95, f"Identical match should be ~1.0, got {score}"

    def test_match_different(self):
        from calibration.gesture_store import CustomGesture
        lm1 = make_landmarks([True, True, False, False, False])
        lm2 = make_landmarks([False, False, True, True, True])
        g = CustomGesture(name="G1", action="click", template=lm1.tolist())
        score = g.match(lm2)
        assert score < 0.6, f"Different gestures should score low, got {score}"


# ── Tests: GestureRecognizer ──────────────────────────────────────────────────

class TestGestureRecognizer:

    def setup_method(self):
        from config.settings import Settings
        from calibration.gesture_store import GestureStore
        from core.gesture_recognizer import GestureRecognizer

        self._tmpdir = tempfile.TemporaryDirectory()
        import config.settings as cs
        self._orig = cs.GESTURES_FILE
        cs.GESTURES_FILE = Path(self._tmpdir.name) / "g.json"

        self.settings = Settings()
        self.settings.gesture_confirm_frames = 1   # Instant for tests
        self.settings.gesture_cooldown_ms = 0
        self.store = GestureStore()
        self.recognizer = GestureRecognizer(self.settings, self.store)

    def teardown_method(self):
        import config.settings as cs
        cs.GESTURES_FILE = self._orig
        self._tmpdir.cleanup()

    def _recognize(self, finger_states, finger_count=None, pinch=False, pinch_dist=0.15):
        from core.hand_tracker import HandData
        hd = make_hand_data(finger_states, finger_count)
        hd.is_pinching = pinch
        hd.pinch_distance = pinch_dist
        return self.recognizer.recognize(hd)

    def test_move_cursor_one_finger(self):
        from core.gesture_recognizer import GestureType
        result = self._recognize([False, True, False, False, False])
        assert result.gesture == GestureType.MOVE_CURSOR

    def test_right_click_two_fingers(self):
        from core.gesture_recognizer import GestureType
        result = self._recognize([False, True, True, False, False])
        assert result.gesture == GestureType.RIGHT_CLICK

    def test_left_click_four_fingers(self):
        from core.gesture_recognizer import GestureType
        result = self._recognize([False, True, True, True, True])
        assert result.gesture == GestureType.LEFT_CLICK

    def test_pinch(self):
        from core.gesture_recognizer import GestureType
        result = self._recognize([True, True, False, False, False],
                                  pinch=True, pinch_dist=0.03)
        assert result.gesture == GestureType.PINCH_CLICK

    def test_fist_no_gesture_initially(self):
        from core.gesture_recognizer import GestureType
        # Fist needs to be held for drag_hold_frames before DRAG fires
        self.settings.drag_hold_frames = 5
        self.recognizer.reset()
        for _ in range(3):
            result = self._recognize([False, False, False, False, False], finger_count=0)
        # Before hold threshold, should not be DRAG yet
        assert result.gesture in (GestureType.NONE, GestureType.DRAG)

    def test_drag_after_hold(self):
        from core.gesture_recognizer import GestureType
        self.settings.drag_hold_frames = 3
        self.recognizer.reset()
        result = None
        for _ in range(5):
            result = self._recognize([False, False, False, False, False], finger_count=0)
        assert result.gesture == GestureType.DRAG

    def test_confirmation_delay(self):
        from core.gesture_recognizer import GestureType
        self.settings.gesture_confirm_frames = 4
        self.settings.gesture_cooldown_ms = 0
        self.recognizer.reset()

        # First 3 frames should return NONE (not confirmed yet)
        for i in range(3):
            result = self._recognize([False, True, True, False, False])
            assert result.gesture == GestureType.NONE, f"Frame {i}: should not confirm yet"

        # 4th frame should confirm
        result = self._recognize([False, True, True, False, False])
        assert result.gesture == GestureType.RIGHT_CLICK

    def test_reset_clears_state(self):
        from core.gesture_recognizer import GestureType
        self.settings.drag_hold_frames = 2
        for _ in range(3):
            self._recognize([False, False, False, False, False], finger_count=0)
        self.recognizer.reset()
        assert not self.recognizer.is_dragging


# ── Tests: CursorController ───────────────────────────────────────────────────

class TestCursorController:

    def setup_method(self):
        from config.settings import Settings
        self.settings = Settings()
        self.settings.smoothing_factor = 0.0   # No smoothing for predictable tests
        self.settings.movement_threshold = 0.0  # Always move

    @patch("pyautogui.moveTo")
    @patch("pyautogui.position", return_value=(320, 240))
    @patch("pyautogui.size", return_value=(1920, 1080))
    def test_move_maps_to_screen(self, mock_size, mock_pos, mock_move):
        from core.cursor_controller import CursorController
        ctrl = CursorController(self.settings, 640, 480)
        ctrl.move((320, 240))   # Center of frame → near center of screen
        assert mock_move.called

    @patch("pyautogui.click")
    @patch("pyautogui.size", return_value=(1920, 1080))
    def test_left_click(self, mock_size, mock_click):
        from core.cursor_controller import CursorController
        ctrl = CursorController(self.settings, 640, 480)
        ctrl.click_left()
        mock_click.assert_called_with(button="left")

    @patch("pyautogui.mouseDown")
    @patch("pyautogui.mouseUp")
    @patch("pyautogui.size", return_value=(1920, 1080))
    def test_drag_start_stop(self, mock_size, mock_up, mock_down):
        from core.cursor_controller import CursorController
        ctrl = CursorController(self.settings, 640, 480)
        ctrl.start_drag()
        assert ctrl.is_dragging
        mock_down.assert_called_with(button="left")

        ctrl.stop_drag()
        assert not ctrl.is_dragging
        mock_up.assert_called_with(button="left")


# ── Tests: CalibrationManager ─────────────────────────────────────────────────

class TestCalibrationManager:

    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        import config.settings as cs
        self._orig = cs.GESTURES_FILE
        cs.GESTURES_FILE = Path(self._tmpdir.name) / "g.json"

        from calibration.gesture_store import GestureStore
        from calibration.calibration_manager import CalibrationManager
        self.store = GestureStore()
        self.mgr = CalibrationManager(self.store)

    def teardown_method(self):
        import config.settings as cs
        cs.GESTURES_FILE = self._orig
        self._tmpdir.cleanup()

    def test_start_sets_active(self):
        self.mgr.start("Test", "left_click")
        assert self.mgr.is_active

    def test_cancel(self):
        self.mgr.start("Test", "left_click")
        self.mgr.cancel()
        assert not self.mgr.is_active

    def test_full_recording_flow(self):
        from calibration.calibration_manager import SAMPLE_COUNT
        completed = []
        self.mgr.start("FlowTest", "left_click",
                        on_complete=lambda g: completed.append(g))

        # Simulate confirm_ready
        self.mgr.confirm_ready()

        # Force to RECORDING state
        from calibration.calibration_manager import CalibrationState
        self.mgr.session.state = CalibrationState.RECORDING
        self.mgr.session.samples = []

        # Feed enough frames
        hd = make_hand_data([False, True, False, False, False])
        for _ in range(SAMPLE_COUNT):
            self.mgr.process_frame(hd)

        assert len(completed) == 1
        assert completed[0].name == "FlowTest"
        assert len(self.store) == 1


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
