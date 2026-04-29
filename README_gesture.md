# Gesture-Controlled Anki

Control Anki flashcard reviews using hand gestures — no keyboard needed. Uses MediaPipe hand landmarks and pure geometric analysis (no ML) with a per-user calibration system.

## What it does

- Detects hand landmarks in real time via webcam
- Classifies gestures using geometric ratios between landmarks (no trained model)
- Maps gestures to Anki keyboard shortcuts
- Displays live Anki deck stats (due, learning, new) via AnkiConnect
- Confidence-based hold bar prevents accidental triggers
- Per-user calibration handles different hand sizes and camera distances

## Gestures

| Gesture | Anki Action | Key |
|---------|-------------|-----|
| Open hand (all fingers extended) | Show answer | Space |
| Thumbs out, fingers closed | Again (1) | 1 |
| Index finger only | Hard (3) | 3 |
| Index + middle (peace) | Easy (4) | 4 |

## How it works

### Landmark geometry
MediaPipe provides 21 hand landmarks per frame. For each finger, the program computes:
- `tip_d` — distance from fingertip to wrist (landmark 0)
- `mcp_d` — distance from MCP knuckle to wrist

A finger is considered **extended** if `tip_d / mcp_d > threshold` (ratio ~1.6 when extended, ~0.9 when closed), or if the fingertip is substantially closer to the camera than the knuckle (depth check).

### Calibration
Press `C` to calibrate. The system collects 45 frames of a closed fist, then 45 of an open hand. It computes the average ratio for each state and sets the detection threshold at the midpoint. This normalises for hand size and camera distance.

### Gesture triggering
A gesture must be held continuously for 1 second before it fires, with a 1.5-second cooldown between triggers. This prevents accidental activation.

### Anki stats
Uses the AnkiConnect API (localhost:8765) to fetch due/learning/new card counts for the active deck, updated every 2 seconds on a background thread.

## Setup

```bash
pip install opencv-python mediapipe pynput requests
```

Requires [AnkiConnect](https://ankiweb.net/shared/info/2055492159) add-on installed in Anki.

## Usage

```bash
python gesture.py
```

- `C` — start calibration (recommended before first use)
- `Q` — quit

## Files

| File | Description |
|------|-------------|
| `gesture.py` | Main script — all gesture detection, calibration, and Anki control |
| `hand_landmarker.task` | MediaPipe model (downloaded automatically on first run) |
