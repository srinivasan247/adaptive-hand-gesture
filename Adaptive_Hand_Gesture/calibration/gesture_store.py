"""
Custom gesture storage and matching.
Stores user-trained gesture templates and provides similarity matching.
"""
 
import json
import numpy as np
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass, field, asdict
from pathlib import Path
 
from config.settings import GESTURES_FILE
 
logger = logging.getLogger(__name__)
 
# Actions that can be bound to a custom gesture (defined here so both modules can import it)
AVAILABLE_ACTIONS = [
    "left_click",
    "right_click",
    "double_click",
    "scroll_up",
    "scroll_down",
    "screenshot",
    "minimize",
    "maximize",
    "close",
]
 
 
@dataclass
class CustomGesture:
    """A user-trained gesture with its landmark template."""
    name: str                    # Display name, e.g. "Peace sign"
    action: str                  # Action key, e.g. "left_click"
    template: List[List[float]]  # (21, 3) normalized landmarks
    description: str = ""
    created_at: str = ""
 
    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
        lm = landmarks.copy()
        
        # 1. Translate wrist to origin (3D)
        wrist = lm[0].copy()
        lm -= wrist
        
        # 2. Scale uniformly across all axes based on hand span
        mcp = lm[9]
        span = np.linalg.norm(mcp)
        if span > 1e-6:
            lm /= span
            
        # 3. Rotate so wrist->mcp points straight up (0, -1, 0)
        # Using a simple 2D rotation for the XY plane usually suffices for orientation
        angle = np.arctan2(lm[9, 1], lm[9, 0])
        d_theta = -np.pi/2 - angle
        
        c, s = np.cos(d_theta), np.sin(d_theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        lm = np.dot(lm, R.T)
        
        return lm

    def match(self, landmarks: np.ndarray) -> float:
        """
        Compute similarity score [0, 1] between 3D landmarks and this template.
        Uses a robust weighted Euclidean distance model (fingertips weighted heavier).
        """
        template_np = np.array(self.template, dtype=np.float32)
        
        query_norm = self.normalize_landmarks(landmarks)
        
        tmpl_norm = getattr(self, '_cached_norm', None)
        if tmpl_norm is None:
            tmpl_norm = self.normalize_landmarks(template_np)
            self._cached_norm = tmpl_norm

        # Weigh fingertips higher (index 4, 8, 12, 16, 20)
        weights = np.ones(21, dtype=np.float32)
        weights[[4, 8, 12, 16, 20]] = 2.5
        weights /= weights.sum()

        distances = np.linalg.norm(query_norm - tmpl_norm, axis=1)
        weighted_dist = np.sum(distances * weights)

        # Map distance to similarity using a sharper exponential curve
        # This makes the threshold much stricter and eliminates random false positives
        score = np.exp(-3.5 * weighted_dist)
        return float(score)
 
    def to_dict(self) -> dict:
        d = asdict(self)
        return d
 
    @classmethod
    def from_dict(cls, d: dict) -> "CustomGesture":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
 
 
class GestureStore:
    """
    Persists and retrieves custom gesture templates.
    """
 
    def __init__(self):
        self._gestures: Dict[str, CustomGesture] = {}
        self._load()
 
    def _load(self):
        if GESTURES_FILE.exists():
            try:
                with open(GESTURES_FILE) as f:
                    data = json.load(f)
                for item in data:
                    g = CustomGesture.from_dict(item)
                    self._gestures[g.name] = g
                logger.info(f"Loaded {len(self._gestures)} custom gestures.")
            except Exception as e:
                logger.error(f"Error loading gestures: {e}")
 
    def save(self):
        GESTURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(GESTURES_FILE, "w") as f:
            json.dump([g.to_dict() for g in self._gestures.values()], f, indent=2)
        logger.info(f"Saved {len(self._gestures)} gestures.")
 
    def add(self, gesture: CustomGesture):
        self._gestures[gesture.name] = gesture
        self.save()
        logger.info(f"Added gesture: {gesture.name}")
 
    def remove(self, name: str) -> bool:
        if name in self._gestures:
            del self._gestures[name]
            self.save()
            logger.info(f"Removed gesture: {name}")
            return True
        return False
 
    def get(self, name: str) -> Optional[CustomGesture]:
        return self._gestures.get(name)
 
    def get_all(self) -> List[CustomGesture]:
        return list(self._gestures.values())
 
    def clear(self):
        self._gestures.clear()
        self.save()
 
    def __len__(self):
        return len(self._gestures)