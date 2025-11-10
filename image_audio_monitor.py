"""
Script audio+vision prêt pour VS Code (Run ▶).
- Source visuelle : image, vidéo (bouclée au FPS natif) ou webcam.
- Détection YOLO continue sur le flux (configurable), avec throttle "toutes les N frames".
- Audio : en continu, chunks <= 3 s, ring buffer 5 fichiers (réécriture).
- Transcription Google en parallèle (speech_recognition via speech_recognition.py).
- Mots-clés audio -> sélection persistante d'une classe : l'objet reste affiché
  jusqu'à ce qu'un autre objet soit demandé.
"""

# ============================ CONFIG ============================

# Source visuelle : chemin image/vidéo, "cam" pour la webcam par défaut, ou un index ("0", "1", …).
SOURCE = "cam"   # ex: "test/photo.jpg" | "test/video.mp4" | "cam" | "0"

# Détection YOLO continue sur le flux
AUTO_DETECT           = True       # True pour détecter en continu sur image/vidéo/webcam quand aucune classe n'est "lockée"
AUTO_DETECT_EVERY     = 2          # inférence toutes les N frames (2 = 1 fois sur 2) pour l'affichage général
AUTO_DETECT_CLASSES   = []         # noms de classes à garder en mode général (ex: ["plane","pen"]); [] = toutes
DEVICE                = "auto"     # "auto" -> GPU si dispo (0), sinon "cpu" | ou "0"/"1" pour forcer un GPU

# Audio
CHUNK_DURATION_SEC    = 3.0        # max 3 s par fichier (sera clampé à 3.0)
AMBIENT_SEC           = 0.0        # calibration bruit ambiant (0 pour désactiver)
LANGUAGES             = ("fr-FR", "en-US")  # ordre d’essai pour la reco Google
OUTPUT_DIR            = "audio_captures"    # dossier pour le ring buffer audio
RING_SIZE             = 5                   # 5 fichiers max

# Sélection audio persistante (classe dessinée en continu tant qu'on ne change pas)
PERSIST_SELECTED      = True
SELECT_EVERY          = 2          # inférence toutes les N frames pour la classe sélectionnée

# YOLO
MODEL_PATH            = "runs_plane/yolov8n_plane/weights/best.pt"
IMG_SIZE              = 640
MIN_CONFIDENCE        = 0.35       # >= 0.35 recommandé
IOU_THRESHOLD         = 0.45       # NMS

# Affichage
MARGIN_RATIO          = 0.9        # redimensionner la fenêtre dans l’écran

# ======================= FIN CONFIG =============================

import importlib.util
import sys
import time
import unicodedata
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
from ultralytics import YOLO

# ---- chargement helper speech_recognition.py
SPEECH_SCRIPT_MODULE = "_speech_script_module"
SPEECH_SCRIPT_PATH = Path(__file__).resolve().parent / "speech_recognition.py"
_spec = importlib.util.spec_from_file_location(SPEECH_SCRIPT_MODULE, SPEECH_SCRIPT_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Impossible de charger le module speech_recognition depuis {SPEECH_SCRIPT_PATH}")
speech_script = importlib.util.module_from_spec(_spec)
sys.modules[SPEECH_SCRIPT_MODULE] = speech_script
_spec.loader.exec_module(speech_script)
speech_rec = speech_script.import_speech_recognition()

# Constantes internes
MAX_CHUNK_SEC = 3.0

CLASS_SYNONYMS: Dict[str, List[str]] = {
    "plane": ["avion", "airplane", "aircraft", "aeroplane", "jet", "avion de ligne", "avion de chasse"],
    "pen": ["stylo", "plume"],
    "glue": ["colle", "adhesif"],
    "cisors": ["ciseaux", "scissors", "cissor", "cisaille"],
}

# ---------- utilitaires visuels ----------
def is_image(path: str) -> bool:
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))

def is_video(path: str) -> bool:
    return path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"))

def load_image(path: str):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Media introuvable: {path}")
    if not is_image(path):
        raise ValueError(f"Format non supporté pour l'image: {path}")
    frame = cv2.imread(path)
    if frame is None:
        raise ValueError(f"Impossible de charger l'image: {path}")
    return frame

def get_screen_size() -> Tuple[int, int]:
    try:
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080

def best_display_size(screen_w: int, screen_h: int, margin_ratio: float) -> Tuple[int, int]:
    return int(screen_w * margin_ratio), int(screen_h * margin_ratio)

def resize_to_fit(img, max_w: int, max_h: int):
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return img
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    if new_w == max_w and new_h == max_h:
        return resized
    top = (max_h - new_h) // 2
    bottom = max_h - new_h - top
    left = (max_w - new_w) // 2
    right = max_w - new_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

def show_frame(window_name: str, frame, screen_w: int, screen_h: int, margin_ratio: float) -> None:
    max_w, max_h = best_display_size(screen_w, screen_h, margin_ratio)
    display = resize_to_fit(frame, max_w, max_h)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h, w = display.shape[:2]
    cv2.resizeWindow(window_name, w, h)
    cv2.imshow(window_name, display)

# ---------- NLP / mots-clés ----------
def parse_languages(raw: Iterable[str]) -> Tuple[str, ...]:
    langs = [token.strip() for token in raw if token.strip()]
    return tuple(langs) if langs else LANGUAGES

def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    without_accents = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_accents.lower()

def build_keyword_patterns(model_names: Dict[int, str]) -> List[Tuple[str, int, str]]:
    patterns: List[Tuple[str, int, str]] = []
    normalized_name_to_id = {normalize_text(name): idx for idx, name in model_names.items()}
    for class_id, name in model_names.items():
        patterns.append((normalize_text(name), class_id, name))
    for key, synonyms in CLASS_SYNONYMS.items():
        normalized_key = normalize_text(key)
        target_id = normalized_name_to_id.get(normalized_key)
        if target_id is None:
            normalized_synonyms = {normalize_text(s) for s in synonyms}
            for class_id, name in model_names.items():
                n = normalize_text(name)
                if n == normalized_key or n in normalized_synonyms:
                    target_id = class_id; break
        if target_id is None:
            continue
        for term in [key, *synonyms]:
            n = normalize_text(term)
            if n:
                patterns.append((n, target_id, term))
    return patterns

def find_first_keyword(text: str, patterns: Sequence[Tuple[str, int, str]]) -> Optional[Tuple[int, str]]:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return None
    best: Optional[Tuple[int, str, int]] = None
    for norm_term, class_id, term in patterns:
        if not norm_term:
            continue
        m = re.search(rf"\b{re.escape(norm_term)}\b", normalized_text)
        if m:
            idx = m.start()
            if best is None or idx < best[2]:
                best = (class_id, term, idx)
    return None if best is None else (best[0], best[1])

# ---------- Device resolver (GPU si dispo, sinon CPU) ----------
def resolve_device(dev: str) -> str:
    if str(dev).lower() != "auto":
        return str(dev)
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

DEVICE_RESOLVED = resolve_device(DEVICE)
print(f"[YOLO] device={DEVICE_RESOLVED}")

# ---------- YOLO utils ----------
def draw_boxes_filtered(frame_bgr, yolo_result, allowed_ids: Optional[set], min_conf: float, class_names: Dict[int, str]):
    """
    Dessine toutes les boîtes pour les classes autorisées (allowed_ids=None => toutes).
    """
    out = frame_bgr.copy()
    boxes = yolo_result.boxes
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        return out, False

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(cls)

    any_drawn = False
    for i in range(len(cls)):
        cid = int(cls[i])
        if allowed_ids is not None and cid not in allowed_ids:
            continue
        cval = float(conf[i])
        if cval < min_conf:
            continue
        any_drawn = True
        x1, y1, x2, y2 = xyxy[i].astype(int)
        label = class_names.get(cid, f"id{cid}")
        display = f"{label} {cval:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (lw, lh), base = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = y1 - 8
        if ly < lh:
            ly = y1 + lh + 8
        top = max(ly - lh - base - 4, 0)
        bottom = min(ly + base, out.shape[0])
        right = min(x1 + lw + 8, out.shape[1])
        cv2.rectangle(out, (x1, top), (right, bottom), color, cv2.FILLED)
        cv2.putText(out, display, (x1 + 4, min(ly, out.shape[0] - base - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out, any_drawn

def yolo_predict_on_frame(model: YOLO, frame_bgr, min_conf: float):
    return model.predict(
        source=frame_bgr, imgsz=IMG_SIZE, conf=min_conf, iou=IOU_THRESHOLD,
        save=False, stream=False, verbose=False, device=DEVICE_RESOLVED
    )[0]

# ---------- affichage / capture visuelle ----------
def get_screen_w_h():
    return get_screen_size()

class VisualLoop(threading.Thread):
    def __init__(self, source: str, window_name: str, margin: float, stop_event: threading.Event,
                 model: YOLO, class_names: Dict[int, str], min_conf: float,
                 auto_detect: bool, detect_every: int, allowed_ids: Optional[set]):
        super().__init__(daemon=True)
        self.source = str(source)
        self.window_name = window_name
        self.margin = margin
        self.stop_event = stop_event

        self.model = model
        self.class_names = class_names
        self.min_conf = min_conf

        self.auto_detect = auto_detect
        self.detect_every = max(1, int(detect_every))
        self.allowed_ids = allowed_ids

        self.screen_w, self.screen_h = get_screen_w_h()

        self.lock = threading.Lock()
        self.latest_raw_frame = None
        self.auto_frame = None               # buffer d'affichage courant (général ou sélection)
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.is_image_source = False
        self.frame_idx = 0

        # Sélection persistante
        self.selected_class_id = None
        self.select_every = max(1, int(SELECT_EVERY))
        self.select_frame_idx = 0

        # init source
        if self.source.lower() in ("cam", "camera"):
            self.cap = cv2.VideoCapture(0)
        elif self.source.isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
        elif is_video(self.source):
            if not Path(self.source).is_file():
                raise FileNotFoundError(f"Vidéo introuvable: {self.source}")
            self.cap = cv2.VideoCapture(self.source)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps and fps > 0 else 30.0
        elif is_image(self.source):
            self.is_image_source = True
            self.latest_raw_frame = load_image(self.source)
            # optionnel: détection 1x sur image si AUTO_DETECT
            if self.auto_detect and self.latest_raw_frame is not None:
                yres = yolo_predict_on_frame(self.model, self.latest_raw_frame, self.min_conf)
                drawn, _ = draw_boxes_filtered(self.latest_raw_frame, yres, self.allowed_ids, self.min_conf, self.class_names)
                self.auto_frame = drawn
        else:
            raise ValueError(f"Source non supportée: {self.source}")

    def set_selected_class(self, class_id: Optional[int]):
        with self.lock:
            self.selected_class_id = None if class_id is None else int(class_id)
            self.select_frame_idx = 0
            # on ne "clear" pas auto_frame : le prochain passage remplacera

    def get_latest_snapshot(self):
        with self.lock:
            return None if self.latest_raw_frame is None else self.latest_raw_frame.copy()

    def run(self):
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            while not self.stop_event.is_set():
                if self.is_image_source:
                    with self.lock:
                        to_show = self.auto_frame if self.auto_frame is not None else self.latest_raw_frame
                    if to_show is not None:
                        show_frame(self.window_name, to_show, self.screen_w, self.screen_h, self.margin)
                    key = cv2.waitKey(30) & 0xFF
                    if key in (27, ord("q")):
                        self.stop_event.set()
                        break
                    continue

                ok, frame = self.cap.read()
                if not ok or frame is None:
                    # fin de fichier -> boucle
                    if self.cap and not self.source.isdigit() and self.source.lower() not in ("cam", "camera"):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        time.sleep(0.01)
                        continue

                self.frame_idx += 1

                # Sélection persistante ?
                with self.lock:
                    sel_id = self.selected_class_id

                if sel_id is not None and PERSIST_SELECTED:
                    self.select_frame_idx += 1
                    if (self.select_frame_idx % self.select_every) == 0:
                        yres = yolo_predict_on_frame(self.model, frame, self.min_conf)
                        # dessine uniquement la classe sel_id
                        out = frame.copy()
                        boxes = yres.boxes
                        any_drawn = False
                        if boxes is not None and boxes.cls is not None and len(boxes) > 0:
                            xyxy = boxes.xyxy.cpu().numpy()
                            cls = boxes.cls.cpu().numpy()
                            conf = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(cls)
                            for i in range(len(cls)):
                                if int(cls[i]) != sel_id or float(conf[i]) < self.min_conf:
                                    continue
                                any_drawn = True
                                x1, y1, x2, y2 = xyxy[i].astype(int)
                                label = self.class_names.get(sel_id, f"id{sel_id}")
                                disp = f"{label} {float(conf[i]):.2f}"
                                color = (0, 255, 0)
                                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                                (lw, lh), base = cv2.getTextSize(disp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                ly = y1 - 8 if y1 - 8 >= lh else y1 + lh + 8
                                top = max(ly - lh - base - 4, 0); bottom = min(ly + base, out.shape[0])
                                right = min(x1 + lw + 8, out.shape[1])
                                cv2.rectangle(out, (x1, top), (right, bottom), color, cv2.FILLED)
                                cv2.putText(out, disp, (x1 + 4, min(ly, out.shape[0] - base - 2)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
                        if not any_drawn:
                            label = self.class_names.get(sel_id, f"id{sel_id}")
                            cv2.putText(out, f"Aucune detection pour {label}", (16, 32),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                        with self.lock:
                            self.auto_frame = out
                else:
                    # pas de sélection -> détection continue normale (throttle)
                    if self.auto_detect and (self.frame_idx % self.detect_every == 0):
                        yres = yolo_predict_on_frame(self.model, frame, self.min_conf)
                        drawn, _ = draw_boxes_filtered(frame, yres, self.allowed_ids, self.min_conf, self.class_names)
                        with self.lock:
                            self.auto_frame = drawn

                with self.lock:
                    self.latest_raw_frame = frame
                    to_show = self.auto_frame if self.auto_frame is not None else frame

                show_frame(self.window_name, to_show, self.screen_w, self.screen_h, self.margin)

                wait_ms = int(1000.0 / max(1.0, self.fps))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (27, ord("q")):
                    self.stop_event.set()
                    break
        finally:
            if self.cap is not None:
                self.cap.release()
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass

# ---------- programme principal ----------
def main():
    # clamp durée chunk
    chunk_sec = min(float(CHUNK_DURATION_SEC), MAX_CHUNK_SEC)
    if chunk_sec <= 0:
        raise ValueError("CHUNK_DURATION_SEC doit être > 0.")
    if not (0.0 < MARGIN_RATIO <= 1.0):
        raise ValueError("MARGIN_RATIO doit être dans (0, 1].")

    # YOLO (device choisi via DEVICE_RESOLVED)
    model = YOLO(MODEL_PATH)

    # noms de classes
    class_names = model.names if hasattr(model, "names") else getattr(model.model, "names", {})
    keyword_patterns = build_keyword_patterns(class_names) if class_names else []
    if not keyword_patterns:
        print("Aucun mot-clé ne correspond aux classes du modèle chargé.")

    min_conf = max(float(MIN_CONFIDENCE), 0.0)

    # map noms -> ids pour filtrage AUTO_DETECT_CLASSES
    allowed_ids = None
    if AUTO_DETECT and AUTO_DETECT_CLASSES:
        inv = {v: k for k, v in class_names.items()}  # name -> id
        sel = set()
        for n in AUTO_DETECT_CLASSES:
            if n in inv:
                sel.add(int(inv[n]))
            else:
                print(f"[AUTO_DETECT] Classe inconnue ignorée: {n}")
        allowed_ids = sel

    # Audio
    out_dir = Path(OUTPUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    languages = parse_languages(LANGUAGES)
    recognizer = speech_rec.Recognizer()
    microphone = speech_rec.Microphone()

    # UI visuelle
    window_name = "Aperçu média"
    stop_event = threading.Event()
    visual = VisualLoop(
        str(SOURCE), window_name, float(MARGIN_RATIO), stop_event,
        model=model, class_names=class_names, min_conf=min_conf,
        auto_detect=AUTO_DETECT, detect_every=AUTO_DETECT_EVERY, allowed_ids=allowed_ids
    )
    visual.start()

    print("Fenêtre active. Appuyez sur 'q' ou 'Echap' (dans la fenêtre) pour arrêter.")
    print(f"Audio continu: chunks de {chunk_sec:.1f}s | Ring buffer: {RING_SIZE} fichiers -> {out_dir}")
    if AUTO_DETECT:
        print(f"Détection continue: every {AUTO_DETECT_EVERY} frames | device={DEVICE_RESOLVED} | filtre={AUTO_DETECT_CLASSES or 'toutes'}")

    # Pool transcription
    executor = ThreadPoolExecutor(max_workers=3)

    def process_chunk_async(audio_data, slot_index: int, wav_path: Path):
        try:
            result = speech_script.transcribe_audio(recognizer, audio_data, languages)
            if result is None:
                print(f"[slot {slot_index}] Pas de transcription exploitable.")
                return
            transcript, confidence, language = result

            # Cherche la 1re classe mentionnée
            keyword = find_first_keyword(transcript, keyword_patterns)
            if keyword is None:
                print(f"[slot {slot_index}] Aucun mot clé. (ASR: {language}, conf {confidence:.2f})")
                return

            target_id, term = keyword
            label = class_names.get(target_id, f"classe {target_id}")

            # Active le mode "sélection persistante" sur la classe demandée
            visual.set_selected_class(target_id)
            print(f"[slot {slot_index}] Sélection persistante -> {label} (demande: '{term}', ASR: {language}, conf {confidence:.2f})")

        except Exception as e:
            print(f"[slot {slot_index}] Erreur transcription/analyse: {e}")

    try:
        with microphone as source:
            if AMBIENT_SEC > 0:
                print(f"Calibration du bruit ambiant ({AMBIENT_SEC:.1f}s)...")
                recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_SEC)
                print("Calibration terminée.")

            iteration = 0
            while not stop_event.is_set():
                slot = iteration % RING_SIZE
                print(f"[{iteration+1}] Capture audio (slot {slot})…")
                audio = recognizer.record(source, duration=chunk_sec)  # bloque le temps du chunk

                wav_path = out_dir / f"capture_{slot}.wav"
                wav_path.write_bytes(audio.get_wav_data())
                print(f"[{iteration+1}] Audio enregistré: {wav_path} (~{chunk_sec:.1f}s)")

                executor.submit(process_chunk_async, audio, slot, wav_path)
                iteration += 1

    except KeyboardInterrupt:
        print("Interruption clavier détectée.")
    finally:
        stop_event.set()
        executor.shutdown(wait=False, cancel_futures=False)

if __name__ == "__main__":
    main()
