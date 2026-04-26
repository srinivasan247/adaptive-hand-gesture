#!/usr/bin/env python3
"""
Quick launcher - run from project root.
Usage:
    python run.py           # Normal launch
    python run.py calibrate # Open calibration on start
    python run.py --no-voice
"""

import sys
import os

# Ensure we're running from the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from main import main
main()
