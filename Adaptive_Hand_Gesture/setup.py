from setuptools import setup, find_packages

setup(
    name="adaptive-gesture",
    version="1.0.0",
    description="Personalized Gesture-Based Cursor Control System",
    author="AdaptiveGesture Project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "pyautogui>=0.9.54",
        "numpy>=1.24.0",
        "SpeechRecognition>=3.10.0",
    ],
    extras_require={
        "voice": ["PyAudio>=0.2.13"],
    },
    entry_points={
        "console_scripts": [
            "adaptive-gesture=main:main",
        ]
    },
)
