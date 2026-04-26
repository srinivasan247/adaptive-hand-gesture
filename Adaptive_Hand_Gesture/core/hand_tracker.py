"""
Real-time hand tracking using MediaPipe.
Detects 21 hand landmarks and exposes finger state analysis.

Supports both MediaPipe legacy API (0.9.x) and new Tasks API (0.10+).
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# Detect which MediaPipe API is available
_USE_TASKS_API = not hasattr(mp, "solutions")
logger.info(f"MediaPipe API: {'Tasks (0.10+)' if _USE_TASKS_API else 'Solutions (legacy)'}")


# MediaPipe landmark indices
class LM:
    WRIST = 0
    THUMB_CMC = 1; THUMB_MCP = 2; THUMB_IP = 3; THUMB_TIP = 4
    INDEX_MCP = 5; INDEX_PIP = 6; INDEX_DIP = 7; INDEX_TIP = 8
    MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
    RING_MCP = 13; RING_PIP = 14; RING_DIP = 15; RING_TIP = 16
    PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20


@dataclass
class HandData:
    """Processed hand information for one frame."""
    landmarks: np.ndarray                 # (21, 3) normalized coords
    landmarks_px: np.ndarray              # (21, 2) pixel coords
    handedness: str                       # "Left" or "Right"
    fingers_up: List[bool]                # [thumb, index, middle, ring, pinky]
    finger_count: int
    index_tip: Tuple[int, int]            # pixel position
    thumb_tip: Tuple[int, int]
    wrist: Tuple[int, int]
    pinch_distance: float                 # normalized 0-1
    pinch_dist_px: float                  # pixel distance (index-thumb)
    mid_pinch_dist_px: float              # pixel distance (index-middle)
    is_pinching: bool
    hand_bbox: Tuple[int, int, int, int]  # x, y, w, h in pixels
    raw_landmarks: any = field(default=None, repr=False)


# ── Hand landmark connections for manual drawing (Tasks API fallback) ─────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


class HandTracker:
    """
    Wraps MediaPipe Hands for real-time hand tracking.
    Automatically uses the correct API based on installed MediaPipe version.
    """

    def __init__(self, settings, frame_w: int = 640, frame_h: int = 480):
        self.settings = settings
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._use_tasks = _USE_TASKS_API

        if self._use_tasks:
            self._init_tasks_api()
        else:
            self._init_solutions_api()

        logger.info("HandTracker initialized.")

    # ── Initializers ──────────────────────────────────────────────────────────

    def _init_solutions_api(self):
        """MediaPipe <= 0.9 / solutions API."""
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.settings.max_hands,
            min_detection_confidence=self.settings.detection_confidence,
            min_tracking_confidence=self.settings.tracking_confidence,
        )

    def _init_tasks_api(self):
        """MediaPipe >= 0.10 Tasks API."""
        import os
        import urllib.request

        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        # Download the hand landmarker model if not present
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "hand_landmarker.task"
        )
        if not os.path.exists(model_path):
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
            logger.info(f"Downloading hand landmarker model to {model_path} ...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
                logger.info("Model downloaded.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download MediaPipe hand landmarker model: {e}\n"
                    f"Please download it manually from:\n{model_url}\n"
                    f"and place it at: {model_path}"
                )

        base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self.settings.max_hands,
            min_hand_detection_confidence=self.settings.detection_confidence,
            min_hand_presence_confidence=self.settings.detection_confidence,
            min_tracking_confidence=self.settings.tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._frame_ts = 0  # millisecond timestamp counter
        self._mp_vision = mp_vision
        self._mp_image = mp.Image

    # ── Main process method ───────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Optional[HandData]:
        """Process a BGR frame and return HandData if a hand is detected."""
        if self._use_tasks:
            return self._process_tasks(frame)
        else:
            return self._process_solutions(frame)

    def _process_solutions(self, frame: np.ndarray) -> Optional[HandData]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None

        hand_lms = results.multi_hand_landmarks[0]
        handedness = "Right"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        lm_array = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32
        )
        lm_px = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in hand_lms.landmark],
            dtype=np.int32,
        )
        return self._build_hand_data(lm_array, lm_px, handedness, hand_lms, w, h)

    def _process_tasks(self, frame: np.ndarray) -> Optional[HandData]:
        h, w = frame.shape[:2]
        self._frame_ts += 33  # ~30 fps in ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = self._mp_image(
            image_format=mp.ImageFormat.SRGB, data=rgb
        )
        results = self._landmarker.detect_for_video(mp_img, self._frame_ts)

        if not results.hand_landmarks:
            return None

        raw_lms = results.hand_landmarks[0]
        handedness = "Right"
        if results.handedness:
            handedness = results.handedness[0][0].display_name

        lm_array = np.array(
            [[lm.x, lm.y, lm.z] for lm in raw_lms], dtype=np.float32
        )
        lm_px = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in raw_lms],
            dtype=np.int32,
        )
        return self._build_hand_data(lm_array, lm_px, handedness, raw_lms, w, h)

    # ── Shared builder ────────────────────────────────────────────────────────

    def _build_hand_data(self, lm_array, lm_px, handedness, raw_lms, w, h) -> HandData:
        fingers_up = self._get_fingers_up(lm_array, handedness)
        finger_count = sum(fingers_up)

        index_tip = (int(lm_px[LM.INDEX_TIP][0]), int(lm_px[LM.INDEX_TIP][1]))
        thumb_tip = (int(lm_px[LM.THUMB_TIP][0]), int(lm_px[LM.THUMB_TIP][1]))
        wrist     = (int(lm_px[LM.WRIST][0]),     int(lm_px[LM.WRIST][1]))

        pinch_dist = float(
            np.linalg.norm(lm_array[LM.THUMB_TIP, :2] - lm_array[LM.INDEX_TIP, :2])
        )
        # Pixel-based distances as used in test_cursor_control.py
        p_idx = np.array(lm_px[LM.INDEX_TIP])
        p_thumb = np.array(lm_px[LM.THUMB_TIP])
        p_mid = np.array(lm_px[LM.MIDDLE_TIP])
        
        pinch_dist_px = float(np.linalg.norm(p_idx - p_thumb))
        mid_pinch_dist_px = float(np.linalg.norm(p_idx - p_mid))

        is_pinching = pinch_dist < self.settings.pinch_threshold

        xs, ys = lm_px[:, 0], lm_px[:, 1]
        x1 = max(0, int(xs.min()) - 15)
        y1 = max(0, int(ys.min()) - 15)
        x2 = min(w, int(xs.max()) + 15)
        y2 = min(h, int(ys.max()) + 15)
        bbox = (x1, y1, x2 - x1, y2 - y1)

        return HandData(
            landmarks=lm_array,
            landmarks_px=lm_px,
            handedness=handedness,
            fingers_up=fingers_up,
            finger_count=finger_count,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            wrist=wrist,
            pinch_distance=pinch_dist,
            pinch_dist_px=pinch_dist_px,
            mid_pinch_dist_px=mid_pinch_dist_px,
            is_pinching=is_pinching,
            hand_bbox=bbox,
            raw_landmarks=raw_lms,
        )

    # ── Finger detection ──────────────────────────────────────────────────────

    def _get_fingers_up(self, lm: np.ndarray, handedness: str) -> List[bool]:
        """
        Determine which fingers are extended.
        Returns [thumb, index, middle, ring, pinky].

        Thumb fix:
          The webcam frame is flipped (cv2.flip) for display, but MediaPipe
          landmarks are computed on the ORIGINAL un-flipped frame.
          This means the handedness label ("Right"/"Left") refers to the
          original frame — which is the OPPOSITE of what the user sees.
          A user holding up their RIGHT hand appears as a LEFT hand in the
          original frame, so MediaPipe labels it "Left".

          Instead of relying on the handedness label (which is confusing),
          we determine thumb side from the hand's own geometry:
          - Find which side of the hand the thumb is on relative to the
            index-MCP → pinky-MCP axis.
          - If thumb tip is on the LEFT of that axis in landmark space,
            the thumb is extended (for the standard webcam mirror view).

          Simplest reliable rule: compare thumb tip x to wrist x.
          In the original (un-flipped) frame:
            - If thumb tip x < wrist x  → thumb is to the LEFT of wrist
              → this is a RIGHT hand (original frame) with thumb extended
              BUT since the frame is flipped for display, this Right hand
              in original = Left hand on screen.
          
          Easiest geometry-based approach that works regardless of flip:
          Use the cross product of (wrist→index_mcp) and (wrist→thumb_tip)
          to determine which side the thumb is on, then check if the
          thumb tip is far enough from the thumb MCP.
        """
        fingers = []

        # ── Thumb: geometry-based, flip-independent ───────────────────────
        # Step 1: Is the thumb OPEN (tip far from index finger base)?
        # Use 3D distance between thumb tip and index MCP as extension signal.
        # When thumb is folded, tip is close to index MCP.
        # When thumb is open/up, tip is far from index MCP.
        thumb_tip   = lm[LM.THUMB_TIP,  :2]
        thumb_ip    = lm[LM.THUMB_IP,   :2]
        thumb_mcp   = lm[LM.THUMB_MCP,  :2]
        index_mcp   = lm[LM.INDEX_MCP,  :2]
        wrist       = lm[LM.WRIST,      :2]

        # Distance from thumb tip to index MCP (normalised coords)
        tip_to_index_mcp = float(np.linalg.norm(thumb_tip - index_mcp))

        # Distance from thumb MCP to index MCP (hand size reference)
        hand_width = float(np.linalg.norm(thumb_mcp - index_mcp)) + 1e-6

        # Ratio > 0.8 means thumb tip is at least 80% of hand-width away
        # from index MCP → thumb is clearly extended / abducted
        thumb_ratio = tip_to_index_mcp / hand_width
        thumb_up = thumb_ratio > 0.8

        fingers.append(thumb_up)

        # ── Other fingers: tip.y < pip.y → extended ───────────────────────
        EXTEND_THRESHOLD = 0.04   # normalised units (~20 px in 480p)
        for tip, pip in [
            (LM.INDEX_TIP,  LM.INDEX_PIP),
            (LM.MIDDLE_TIP, LM.MIDDLE_PIP),
            (LM.RING_TIP,   LM.RING_PIP),
            (LM.PINKY_TIP,  LM.PINKY_PIP),
        ]:
            fingers.append(bool(lm[tip][1] < lm[pip][1] - EXTEND_THRESHOLD))

        return fingers

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_landmarks(self, frame: np.ndarray, hand_data: HandData) -> np.ndarray:
        """Draw hand landmarks on frame (works for both API versions)."""
        if hand_data.raw_landmarks is None:
            return frame

        if self._use_tasks:
            self._draw_tasks(frame, hand_data.landmarks_px)
        else:
            self.mp_draw.draw_landmarks(
                frame,
                hand_data.raw_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )
        return frame

    def _draw_tasks(self, frame: np.ndarray, lm_px: np.ndarray):
        """Manually draw landmarks for Tasks API (no drawing_utils available)."""
        # Draw connections
        for a, b in HAND_CONNECTIONS:
            pt1 = (int(lm_px[a][0]), int(lm_px[a][1]))
            pt2 = (int(lm_px[b][0]), int(lm_px[b][1]))
            cv2.line(frame, pt1, pt2, (80, 180, 80), 2)

        # Draw landmark dots
        for i, (x, y) in enumerate(lm_px):
            color = (0, 255, 255) if i in (4, 8, 12, 16, 20) else (255, 255, 255)
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 0), 1)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        if self._use_tasks:
            self._landmarker.close()
        else:
            self.hands.close()
        logger.info("HandTracker closed.")