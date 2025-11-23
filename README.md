# Project Overview

This project groups several scripts around YOLOv8 object detection, with an interactive keyword‑based workflow and an audio+vision demo.

- `predict_keywords.py`
  - Displays an image or video continuously
  - You can type object names at any time in the console (separated by `&`)
  - Bounding boxes appear in different colors for each requested class
  - If a requested object is not detected, a red banner “Non detected: …” is shown at the top
  - Quit: `q`/Esc in the window, or `exit`/`quit` in the console

- `image_audio_monitor.py`
  - Reads from image / video / webcam
  - Periodic YOLO detection (configurable)
  - Audio capture in short chunks (ring buffer)
  - Asynchronous transcription (library `speech_recognition`); audio keywords are used to select an object to display continuously

## Installation

- Python 3.10+ recommended
- Python modules used by the scripts:
  - `ultralytics` (YOLOv8, detection)
  - `torch` (deep‑learning backend, CPU or GPU)
  - `opencv-python` (OpenCV, video / webcam / display)
  - `numpy` (tensor manipulation, used via `torch` / `ultralytics`)
  - `SpeechRecognition` (speech recognition)
  - Microphone backend for `SpeechRecognition`: `pyaudio` **or** `sounddevice`
  - `tkinter` is used to get the screen size (usually included with Python on Windows)

### Recommended installation order

The order below takes advantage of the fact that `ultralytics` automatically installs several useful dependencies (`numpy`, `opencv-python`, etc.).

```bash
# (Optional) Create a clean virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate # Linux/macOS

# 1) Install PyTorch BEFORE ultralytics
# See https://pytorch.org/ for the command adapted to your OS/GPU.
# Example CPU (Windows/Linux):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2) Install YOLOv8 + vision
# ultralytics will automatically install several useful modules (numpy, opencv-python, etc.).
pip install ultralytics

# 3) Complete vision dependencies if needed
# (in case your environment did not receive them)
pip install opencv-python numpy

# 4) Audio modules (for image_audio_monitor.py and speech_recognition.py)
pip install SpeechRecognition
# Microphone backend: choose the one that works best on your machine
pip install pyaudio      # often used on Windows (precompiled wheel recommended)
pip install sounddevice  # cross‑platform alternative
```

## Models

- By default, the scripts point to `runs_plane/yolov8n_plane/weights/best.pt`.
- Adjust `--model` (CLI) or the `MODEL_PATH` constant in `image_audio_monitor.py`.

## Usage – predict_keywords.py

```bash
python predict_keywords.py --model runs_plane/yolov8n_plane/weights/best.pt --source test/IMG_4160.jpg
```

Main options:
- `--model`: path to the YOLOv8 weights
- `--source`: image, video or folder (the first media file in the folder will be used)
- `--targets`: (optional) initial list of objects (FR/EN synonyms supported)
- `--imgsz`: input size (default 640)
- `--conf`: minimum confidence threshold (≥ 0.35 recommended)

Interaction:
- Type object names in the console (e.g. `plane & pen`) then press Enter.
- The window updates automatically.
- Quit: `q`/Esc (window) or `exit`/`quit` (console).

Tip: colors are assigned in the order you type them (object 1: green, object 2: red, etc.).

## Usage – image_audio_monitor.py

- Open the file and adjust the CONFIG section at the top of the script:
  - `SOURCE`: image/video path or `"cam"` / webcam index (`"0"`)
  - `AUTO_DETECT*`: general detection frequency
  - `MODEL_PATH`, `IMG_SIZE`, `MIN_CONFIDENCE`
  - Audio: `CHUNK_DURATION_SEC`, `OUTPUT_DIR`, `RING_SIZE`, `LANGUAGES`

Run:

```bash
python image_audio_monitor.py
```

- The script performs detections and records audio samples continuously.
- Google transcription (via `speech_recognition`) is used to map audio keywords to classes (FR/EN synonyms). The selected object remains displayed until a new keyword is detected.

## Useful files

- `classes.txt`: class mappings (if applicable to your model)
- `data.yaml`: dataset configuration (YOLO)
- Folders:
  - `runs*`: YOLO training outputs (weights)
  - `images/`, `videos/`, `test/`: examples (adjust as needed)

## Troubleshooting

- OpenCV window does not respond: make sure the window has focus. Try `q`/Esc to exit.
- ImportError torch/ultralytics: install/adjust versions according to your OS and GPU (see the PyTorch website).
- Videos do not open: install the required codecs (Windows) or convert to mp4/h264.
- Microphone not detected (audio): check drivers, permissions and backend (`pyaudio` / `sounddevice`).

## License

This repository is provided as‑is for educational/demo use. Adapt as needed.

