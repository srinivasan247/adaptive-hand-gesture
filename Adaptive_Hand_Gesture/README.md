# AdaptiveGesture — Personalized Gesture-Based Cursor Control

A flexible, adaptive system for controlling your computer cursor using hand gestures captured through a webcam. Designed to support users with diverse physical abilities including missing fingers, tremors, or limited hand mobility.

---

## Features

| Feature | Description |
|---|---|
| Real-time hand tracking | 21 landmark detection via MediaPipe at 30+ FPS |
| Gesture-based cursor control | Index finger controls mouse position |
| Built-in gesture set | Click, right-click, drag, scroll, pinch |
| **Personalized calibration** | Train your own custom gestures |
| Cursor stabilization | Smoothing + jitter filtering for tremors |
| Gesture confirmation delay | Prevents accidental triggers |
| Voice command integration | Hybrid gesture + voice interaction |
| Visual feedback overlay | Live landmark display, gesture labels, FPS |
| Accessible design | Works with missing fingers, extra fingers, deformities |

---

## Default Gesture Map

| Gesture | Action |
|---|---|
| ☝️ 1 finger (index only) | Move cursor |
| ✌️ 2 fingers (index + middle) | Right click |
| 🤟 3 fingers | Scroll (move hand up/down) |
| 🖖 4 fingers (no thumb) | Left click |
| ✊ Closed fist (hold) | Start drag |
| 🤏 Pinch (thumb + index) | Click |
| 🖐️ Open palm | Release drag |

---

## Project Structure

```
gesture_control/
├── main.py                        # Entry point
├── requirements.txt
├── setup.py
│
├── config/
│   ├── __init__.py
│   └── settings.py                # All configurable parameters
│
├── core/
│   ├── __init__.py
│   ├── hand_tracker.py            # MediaPipe hand tracking (21 landmarks)
│   ├── gesture_recognizer.py      # Gesture classification + confirmation
│   ├── cursor_controller.py       # Screen mapping, smoothing, drag/click
│   └── action_executor.py         # Translates gestures → system actions
│
├── calibration/
│   ├── __init__.py
│   ├── gesture_store.py           # Save/load custom gesture templates
│   └── calibration_manager.py     # Recording session state machine
│
├── voice/
│   ├── __init__.py
│   └── voice_handler.py           # Background speech recognition thread
│
├── ui/
│   ├── __init__.py
│   ├── app_window.py              # Main loop + subsystem orchestration
│   ├── overlay_renderer.py        # OpenCV HUD drawing
│   ├── calibration_dialog.py      # Tkinter calibration GUI
│   └── settings_dialog.py         # Tkinter settings GUI
│
└── utils/
    ├── __init__.py
    └── logger.py                  # Rotating file + console logging
```

Saved data lives in `~/.adaptive_gesture/`:
```
~/.adaptive_gesture/
├── settings.json      # User preferences
├── gestures.json      # Calibrated custom gestures
└── logs/
    └── gesture_control.log
```

---

## Installation

### 1. Clone / download the project

```bash
git clone https://github.com/yourname/adaptive-gesture.git
cd adaptive-gesture/gesture_control
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### PyAudio (for voice commands) — platform notes:

**Linux:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install PyAudio
```

**macOS:**
```bash
brew install portaudio
pip install PyAudio
```

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

### 4. Tkinter (for dialogs)

Tkinter is included with most Python distributions.  
On Linux, if missing:
```bash
sudo apt-get install python3-tk
```

---

## Running

```bash
# Standard launch
python main.py

# Specify camera index (if you have multiple cameras)
python main.py --camera 1

# Launch directly into calibration mode
python main.py --calibrate

# Disable voice commands
python main.py --no-voice

# Enable debug logging
python main.py --debug
```

---

## Keyboard Controls (in the camera window)

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `P` | Pause / Resume gesture control |
| `C` | Open Calibration Dialog |
| `S` | Open Settings Dialog |
| `SPACE` | Confirm gesture during calibration |
| `R` | Reset gesture recognizer state |
| `H` | Print help to console |

---

## Personalized Calibration

This is the core innovation — train the system to recognize **your** gestures.

### Steps:
1. Press `C` in the camera window
2. Enter a gesture name (e.g., "My Click") and select an action
3. Click **Start Calibration**
4. Position your hand showing the gesture you want to train
5. Press `SPACE` when ready
6. Hold the gesture steady through the 3-2-1 countdown
7. The system records 30 frames and saves the template

The system averages all samples to create a robust template, then uses **normalized Euclidean distance** matching (translation + scale invariant) for recognition.

### Supported actions for custom gestures:
- `left_click`, `right_click`, `double_click`
- `scroll_up`, `scroll_down`
- `screenshot`, `minimize`, `maximize`, `close`

---

## Voice Commands

Speak naturally. Supported phrases:

| Phrase | Action |
|---|---|
| "click" / "left click" | Left click |
| "right click" | Right click |
| "double click" | Double click |
| "scroll up" | Scroll up |
| "scroll down" | Scroll down |
| "minimize" | Minimize window |
| "maximize" | Maximize window |
| "close" / "close window" | Close window |
| "screenshot" | Take screenshot |
| "stop" / "pause" | Pause gesture control |
| "resume" | Resume gesture control |
| "start calibration" | Open calibration dialog |
| "open browser" | Open browser |

---

## Accessibility Notes

### For users with missing fingers:
The system does not hardcode "5 fingers = open palm." It tracks whatever fingers are present. Use calibration to map your available gestures to any action.

### For users with tremors:
- Increase **Smoothing Factor** (0 = raw, 0.95 = maximum smoothing)
- Increase **Movement Threshold** (px) to filter micro-movements
- Increase **Confirm Frames** to require longer gesture holds
- All adjustable in the Settings dialog (`S` key)

### For users with limited mobility:
- Use voice commands as primary input alongside minimal gestures
- Calibrate simple, achievable postures to common actions
- Pinch gesture works with just thumb and index finger

---

## Tuning Tips

| Problem | Solution |
|---|---|
| Cursor too jittery | Increase smoothing factor, increase movement threshold |
| Gestures not recognized | Improve lighting, adjust detection confidence |
| Accidental clicks | Increase confirm frames, increase cooldown ms |
| Drag not triggering | Decrease drag hold frames |
| Custom gesture false positives | Re-calibrate with more consistent samples |
| Voice not working | Check microphone permissions, install PyAudio |

---

## Technical Architecture

```
Camera Frame
    │
    ▼
HandTracker (MediaPipe)
    │ HandData (21 landmarks, finger states)
    ▼
GestureRecognizer
    ├── Built-in rule classifier (finger count + positions)
    ├── Custom gesture matcher (normalized landmark distance)
    └── Confirmation delay (N-frame hold filter)
    │ GestureResult
    ▼
ActionExecutor
    ├── CursorController (pyautogui, exponential smoothing)
    └── System actions (click, scroll, drag, shell commands)

VoiceCommandHandler (background thread)
    └── SpeechRecognition → same ActionExecutor

OverlayRenderer
    └── OpenCV HUD (landmarks, gesture label, FPS, status)

CalibrationManager (state machine)
    └── GestureStore (JSON persistence)
```

---

## License

MIT License — free to use, modify, and distribute.
