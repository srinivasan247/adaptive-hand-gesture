"""
Calibration session - guides user through recording a new custom gesture.
"""
 
import numpy as np
import logging
import time
from enum import Enum, auto
from typing import Optional, List, Callable
from dataclasses import dataclass
 
from core.hand_tracker import HandData
from calibration.gesture_store import GestureStore, CustomGesture, AVAILABLE_ACTIONS
 
logger = logging.getLogger(__name__)
 
 
class CalibrationState(Enum):
    IDLE = auto()
    WAITING_FOR_GESTURE = auto()    # Prompt shown, waiting for user to position
    COUNTDOWN = auto()              # 3-2-1 countdown
    RECORDING = auto()              # Collecting samples
    CONFIRMING = auto()             # Showing average, asking to confirm
    COMPLETE = auto()
    CANCELLED = auto()
 
 
 
 
SAMPLE_COUNT = 30        # Frames to collect per gesture
COUNTDOWN_SECS = 3
 
 
@dataclass
class CalibrationSession:
    """State for a single calibration recording session."""
    gesture_name: str = ""
    gesture_action: str = ""
    state: CalibrationState = CalibrationState.IDLE
    samples: List[np.ndarray] = None
    countdown_start: float = 0.0
    on_complete: Optional[Callable] = None
    on_cancel: Optional[Callable] = None
    status_message: str = ""
    progress: float = 0.0     # 0.0 - 1.0 recording progress
 
    def __post_init__(self):
        if self.samples is None:
            self.samples = []
 
 
class CalibrationManager:
    """
    Manages the calibration workflow.
    Records gesture samples and saves them to GestureStore.
    """
 
    def __init__(self, gesture_store: GestureStore):
        self.gesture_store = gesture_store
        self.session: Optional[CalibrationSession] = None
 
    @property
    def is_active(self) -> bool:
        return (self.session is not None and
                self.session.state not in (CalibrationState.IDLE,
                                           CalibrationState.COMPLETE,
                                           CalibrationState.CANCELLED))
 
    @property
    def state(self) -> CalibrationState:
        if self.session:
            return self.session.state
        return CalibrationState.IDLE
 
    def start(self, gesture_name: str, gesture_action: str,
              on_complete: Callable = None, on_cancel: Callable = None):
        """Begin a calibration session for a new gesture."""
        self.session = CalibrationSession(
            gesture_name=gesture_name,
            gesture_action=gesture_action,
            state=CalibrationState.WAITING_FOR_GESTURE,
            on_complete=on_complete,
            on_cancel=on_cancel,
            status_message=f"Get ready: '{gesture_name}'\nHold your gesture and press SPACE"
        )
        logger.info(f"Calibration started: {gesture_name} -> {gesture_action}")
 
    def confirm_ready(self):
        """User pressed SPACE - begin countdown."""
        if self.session and self.session.state == CalibrationState.WAITING_FOR_GESTURE:
            self.session.state = CalibrationState.COUNTDOWN
            self.session.countdown_start = time.time()
            self.session.status_message = "Get ready..."
 
    def process_frame(self, hand_data: Optional[HandData]) -> str:
        """
        Call each frame during calibration.
        Returns current status message for overlay.
        """
        if not self.session or not self.is_active:
            return ""
 
        s = self.session
 
        if s.state == CalibrationState.WAITING_FOR_GESTURE:
            return s.status_message
 
        if s.state == CalibrationState.COUNTDOWN:
            elapsed = time.time() - s.countdown_start
            remaining = max(0, COUNTDOWN_SECS - elapsed)
            if remaining <= 0:
                s.state = CalibrationState.RECORDING
                s.samples = []
                s.status_message = "Recording..."
            else:
                s.status_message = f"Recording in {int(remaining) + 1}..."
            return s.status_message
 
        if s.state == CalibrationState.RECORDING:
            if hand_data is not None:
                s.samples.append(hand_data.landmarks.copy())
                s.progress = len(s.samples) / SAMPLE_COUNT
                s.status_message = f"Recording... {len(s.samples)}/{SAMPLE_COUNT}"
 
                if len(s.samples) >= SAMPLE_COUNT:
                    self._finalize()
            else:
                s.status_message = "Show your hand to the camera..."
            return s.status_message
 
        if s.state == CalibrationState.CONFIRMING:
            return s.status_message
 
        return s.status_message
 
    def _finalize(self):
        """Average samples and save gesture."""
        s = self.session
        if not s.samples:
            self.cancel()
            return
 
        # Normalize all collected landmark samples before averaging
        norm_samples = []
        for smp in s.samples:
            norm = CustomGesture.normalize_landmarks(smp)
            norm_samples.append(norm)
            
        avg_template = np.mean(norm_samples, axis=0)
        gesture = CustomGesture(
            name=s.gesture_name,
            action=s.gesture_action,
            template=avg_template.tolist(),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.gesture_store.add(gesture)
        s.state = CalibrationState.COMPLETE
        s.status_message = f"✓ Gesture '{s.gesture_name}' saved!"
        logger.info(f"Gesture calibrated and saved: {s.gesture_name}")
 
        if s.on_complete:
            s.on_complete(gesture)
 
    def cancel(self):
        if self.session:
            self.session.state = CalibrationState.CANCELLED
            self.session.status_message = "Calibration cancelled."
            if self.session.on_cancel:
                self.session.on_cancel()
        logger.info("Calibration cancelled.")
 
    def get_progress(self) -> float:
        return self.session.progress if self.session else 0.0
 
    def get_status(self) -> str:
        return self.session.status_message if self.session else ""