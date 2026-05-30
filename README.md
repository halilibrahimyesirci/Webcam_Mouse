# Webcam Mouse — eye-tracking cursor control

Control the mouse cursor with your eyes. Webcam Mouse uses real-time iris tracking to move the
pointer, so you can navigate hands-free. It was built as an accessibility-minded project for the
*Turkcell Yarının Teknolojileri* competition.

## Demo

<!-- Add a short screen recording: drop a GIF into the repo and reference it here, e.g. -->
<!-- ![demo](docs/demo.gif) -->

> A demo GIF will be added here.

## Features

- Real-time iris (pupil) detection with MediaPipe, independent of head position
- Relative gaze mapping from the eye socket to screen coordinates
- Smooth, low-latency cursor movement with adjustable sensitivity
- Three-stage calibration (near / mid / far), five points each, saved to reusable profiles
- Safety bounds and a quick disable shortcut (ESC + L)
- Modern CustomTkinter UI with a live webcam view and a debug / gaze panel
- Mirrored-image support

## Tech stack

Python · MediaPipe · OpenCV · dlib · NumPy · CustomTkinter · screeninfo · Pillow

## Requirements

- Python 3.10+
- A standard webcam (720p or better)

## Setup

```bash
git clone https://github.com/halilibrahimyesirci/Webcam_Mouse
cd Webcam_Mouse
pip install -r requirements.txt
python main.py
```

## How it works

1. The webcam feed is processed frame by frame to locate the eyes and the iris within each socket.
2. The iris position is turned into a relative gaze point and mapped to screen coordinates.
3. After calibration, that point drives the cursor, with smoothing applied to reduce jitter.

Calibration profiles are saved to disk, so you don't have to recalibrate every session.

## License

[GPL-3.0](LICENSE)
