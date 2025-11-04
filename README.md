# Project Overview

Ce projet regroupe plusieurs scripts autour de la détection d’objets YOLOv8, avec un flux d’usage interactif par mots‑clés et une démo audio+vision.

- `predict_keywords.py`
  - Affiche une image/vidéo en continu
  - Vous tapez à tout moment des objets dans la console (séparés par `&`)
  - Les boîtes apparaissent en couleurs différentes par classe demandée
  - Si un objet demandé n’est pas détecté, un bandeau rouge « Non detecte: … » s’affiche en haut
  - Quitter: `q`/Échap dans la fenêtre, ou `exit`/`quit` dans la console

- `image_audio_monitor.py`
  - Lecture image/vidéo/webcam
  - Détection YOLO périodique (configurable)
  - Capture audio par tranches courtes (ring buffer)
  - Transcription asynchrone (lib `speech_recognition`), mots‑clés audio pour sélectionner un objet à afficher en continu

## Installation

- Python 3.10+ recommandé
- Dépendances principales:
  - `ultralytics` (YOLOv8)
  - `opencv-python`
  - `numpy`
  - `torch` (CPU ou GPU selon votre environnement)
  - Pour l’audio (script optionnel `image_audio_monitor.py`): `speechrecognition`, et un backend micro (ex. `pyaudio` ou `sounddevice` selon votre setup)

Commandes typiques:

```bash
pip install ultralytics opencv-python numpy
# Torch: reportez-vous à https://pytorch.org/ pour la commande adaptée à votre OS/GPU
# Ex. CPU (Windows/Linux):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Audio (facultatif pour image_audio_monitor.py)
pip install SpeechRecognition
# backend micro selon votre OS (exemples):
# pip install pyaudio   # Windows/macOS (selon pilotes) ou binaire précompilé
# pip install sounddevice
```

## Modèles

- Par défaut, les scripts pointent sur `runs_plane/yolov8n_plane/weights/best.pt`.
- Adaptez `--model` (CLI) ou la constante `MODEL_PATH` dans `image_audio_monitor.py`.

## Utilisation — predict_keywords.py

```bash
python predict_keywords.py --model runs_plane/yolov8n_plane/weights/best.pt --source test/IMG_4160.jpg
```

Options principales:
- `--model`: chemin vers les poids YOLOv8
- `--source`: image, vidéo ou dossier (le premier média du dossier sera utilisé)
- `--targets`: (optionnel) liste initiale d’objets (synonymes FR/EN pris en charge)
- `--imgsz`: taille d’entrée (défaut 640)
- `--conf`: seuil min de confiance (≥ 0.35 recommandé)

Interaction:
- Tapez des noms d’objets dans la console (ex.: `avion & stylo`) puis Entrée.
- La fenêtre s’actualise automatiquement.
- Quitter: `q`/Échap (fenêtre) ou `exit`/`quit` (console).

Astuce: les couleurs sont affectées par ordre de saisie (objet 1: vert, objet 2: rouge, etc.).

## Utilisation — image_audio_monitor.py

- Ouvrez le fichier et ajustez la section CONFIG en tête de script:
  - `SOURCE`: image/vidéo (chemin) ou `"cam"` / index de webcam ("0")
  - `AUTO_DETECT*`: fréquence de détection générale
  - `MODEL_PATH`, `IMG_SIZE`, `MIN_CONFIDENCE`
  - Audio: `CHUNK_DURATION_SEC`, `OUTPUT_DIR`, `RING_SIZE`, `LANGUAGES`

Exécution:

```bash
python image_audio_monitor.py
```

- Le script effectue des détections et enregistre des échantillons audio en continu.
- La transcription Google (via `speech_recognition`) est utilisée pour mapper des mots‑clés audio aux classes (synonymes FR/EN). L’objet sélectionné reste affiché jusqu’à un nouveau mot.

## Fichiers utiles

- `classes.txt`: correspondances des classes (si applicable à votre modèle)
- `data.yaml`: configuration dataset (YOLO)
- Dossiers:
  - `runs*`: sorties d’entraînement YOLO (poids)
  - `images/`, `videos/`, `test/`: exemples (adaptez si besoin)

## Dépannage

- OpenCV window ne réagit pas: assurez-vous que la fenêtre a le focus. Essayez `q`/Échap pour quitter.
- ImportError torch/ultralytics: installez/ajustez les versions selon votre OS et GPU (voir site PyTorch).
- Vidéos ne s’ouvrent pas: installez les codecs nécessaires (Windows) ou convertissez en mp4/h264.
- Micro non détecté (audio): vérifiez les pilotes, permissions et backend (pyaudio/sounddevice).

## Licence

Ce dépôt est fourni tel quel pour usage pédagogique/démo. Adaptez selon vos besoins.
