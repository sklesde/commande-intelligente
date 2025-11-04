"""
predict_keywords.py
--------------------
Détection d'objets YOLOv8 filtrée par mots-clés avec saisie continue.

Fonctionnement:
- Ouvre une image ou une vidéo et affiche en continu les détections.
- Vous tapez à tout moment des objets dans le terminal (séparés par '&').
- Les boîtes demandées sont dessinées avec des couleurs distinctes.
- Si un objet demandé est absent, un bandeau rouge "Non detecte: ..." s'affiche en haut.
- Quitter: 'q' ou Échap dans la fenêtre, ou 'exit'/'quit' dans la console.

Prérequis: ultralytics, opencv-python, torch (CPU/GPU selon votre environnement).
"""

import argparse
import os
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import cv2
from ultralytics import YOLO
import threading

# ==========================
# User defaults
# ==========================
DEFAULT_MODEL_PATH = "runs_plane/yolov8n_plane/weights/best.pt"
DEFAULT_SOURCE = "test/IMG_4160.jpg"
MIN_CONFIDENCE = 0.35  # Mask detections below 35 %

# Synonyms allow matching English/French variants for common classes.
CLASS_SYNONYMS: Dict[str, List[str]] = {
    "plane": [
        "avion", "airplane", "aircraft", "aeroplane", "jet"],
    "pen": ["stylo", "plume"],
    "glue": ["colle", "adhesif"],
    "cisors": ["ciseaux", "scissors", "cissor", "cisaille"]
}

# ==========================
# Utilities
# ==========================
IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".m4v"]


def is_image(path: str) -> bool:
    return any(path.lower().endswith(ext) for ext in IMAGE_EXT)


def is_video(path: str) -> bool:
    return any(path.lower().endswith(ext) for ext in VIDEO_EXT)


def get_screen_size() -> Tuple[int, int]:
    """Return monitor size; fall back to 1920x1080 when unavailable."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return 1920, 1080


def resize_to_fit(img, max_w: int, max_h: int):
    """Resize image to fit within max_w x max_h while keeping aspect ratio."""
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return img

    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    if new_w == max_w and new_h == max_h:
        return resized

    top = (max_h - new_h) // 2
    bottom = max_h - new_h - top
    left = (max_w - new_w) // 2
    right = max_w - new_w - left

    return cv2.copyMakeBorder(
        resized,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def best_display_size(screen_w: int, screen_h: int, margin_ratio: float = 0.9) -> Tuple[int, int]:
    """Reserve a margin of the screen to avoid overlapping OS elements."""
    return int(screen_w * margin_ratio), int(screen_h * margin_ratio)


def show_frame(window_name: str, frame, screen_w: int, screen_h: int) -> None:
    """Affiche `frame` dans une fenêtre redimensionnée au mieux pour l'écran."""
    max_w, max_h = best_display_size(screen_w, screen_h, margin_ratio=0.9)
    display = resize_to_fit(frame, max_w, max_h)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h, w = display.shape[:2]
    cv2.resizeWindow(window_name, w, h)
    cv2.imshow(window_name, display)


def display_preview(path: str, window_name: str, screen_w: int, screen_h: int) -> bool:
    """Show a preview frame before running detections."""
    if is_image(path):
        frame = cv2.imread(path)
        if frame is None:
            print("Erreur : impossible de charger l'image pour prévisualisation.")
            return False
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("Erreur : impossible d'ouvrir la vidéo pour prévisualisation.")
            return False
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print("Erreur : impossible de lire une frame de prévisualisation.")
            return False

    preview = frame.copy()
    cv2.putText(
        preview,
        "Apercu source - appuyez sur une touche dans la fenetre pour continuer",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    show_frame(window_name, preview, screen_w, screen_h)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return True


def list_known_classes(model_names: Dict[int, str]) -> List[str]:
    """Retourne la liste ordonnée des noms de classes du modèle."""
    return [model_names[idx] for idx in sorted(model_names)]


def print_known_classes(known_names: Sequence[str]) -> None:
    """Affiche les classes connues ainsi que leurs synonymes pris en charge."""
    print("Classes disponibles :")
    for name in known_names:
        synonyms = CLASS_SYNONYMS.get(name.lower(), [])
        if synonyms:
            joined = ", ".join(synonyms)
            print(f" - {name} (synonymes: {joined})")
        else:
            print(f" - {name}")


def prompt_for_targets() -> List[str]:
    """Invite simple: lit une ligne et découpe par '&' et ','."""
    raw = input(
        "Objet(s) a rechercher (utilisez & pour plusieurs, ex: avion & stylo). Tapez 'exit' pour quitter : "
    )
    # Support both '&' and ',' as separators; '&' is recommended per requirements
    tokens: List[str] = []
    for part in raw.split("&"):
        for sub in part.split(","):
            val = sub.strip()
            if val:
                tokens.append(val)
    return tokens


def parse_input_line(raw: str) -> List[str]:
    """Découpe une ligne utilisateur en tokens, séparés par '&' et ','."""
    tokens: List[str] = []
    for part in raw.split("&"):
        for sub in part.split(","):
            val = sub.strip()
            if val:
                tokens.append(val)
    return tokens


def resolve_ordered(words: Sequence[str], model_names: Dict[int, str]) -> Tuple[List[int], List[str]]:
    """Résout les mots vers des IDs de classes, en conservant l'ordre de saisie."""
    lookup = _build_label_lookup(model_names)
    ordered_ids: List[int] = []
    unmatched: List[str] = []
    for w in words:
        base = w.strip().lower()
        variants = {
            base,
            base.replace(" ", ""),
            base.replace(" ", "_"),
            base.replace("_", " "),
            base.replace("_", ""),
            base.replace("-", " "),
            base.replace("-", ""),
        }
        cid = None
        for v in variants:
            if v in lookup:
                cid = lookup[v]
                break
        if cid is None:
            unmatched.append(w)
        else:
            if cid not in ordered_ids:
                ordered_ids.append(cid)
    return ordered_ids, unmatched


class TargetState:
    """État partagé entre le thread d'entrée console et l'affichage.

    - ordered_ids: classes choisies (dans l'ordre saisi)
    - unmatched: mots non reconnus lors de la dernière saisie
    - updated: indicateur de mise à jour (force un refresh)
    - lock: protège l'accès concurrent à l'état
    """
    def __init__(self) -> None:
        self.ordered_ids: List[int] = []
        self.unmatched: List[str] = []
        self.updated: bool = True
        self.lock = threading.Lock()


def build_color_map(ordered_ids: List[int]) -> Dict[int, Tuple[int, int, int]]:
    """Construit une palette couleur par classe, stable selon l'ordre saisi."""
    palette: List[Tuple[int, int, int]] = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 128, 255),
        (128, 0, 255),
        (255, 0, 128),
        (0, 200, 200),
        (200, 200, 0),
    ]
    cmap: Dict[int, Tuple[int, int, int]] = {}
    for i, cid in enumerate(ordered_ids):
        cmap[cid] = palette[i % len(palette)]
    return cmap


def input_loop(state: 'TargetState', model_names: Dict[int, str], stop_event: 'threading.Event') -> None:
    """Lit en continu la saisie utilisateur dans le terminal et met à jour l'état."""
    print("Saisissez des objets (ex: avion \u0026 stylo). 'exit' pour quitter.")
    while not stop_event.is_set():
        try:
            raw = input(">> ")
        except EOFError:
            stop_event.set()
            break
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            stop_event.set()
            break
        words = parse_input_line(line)
        ordered_ids, unmatched = resolve_ordered(words, model_names)
        with state.lock:
            state.ordered_ids = ordered_ids
            state.unmatched = unmatched
            state.updated = True
        if unmatched:
            print("Mots non reconnus :", ", ".join(unmatched))
        if ordered_ids:
            print("Recherche des classes :", ", ".join(model_names[i] for i in ordered_ids))


def _build_label_lookup(model_names: Dict[int, str]) -> Dict[str, int]:
    """Create a lookup from normalized label/synonym variants to class id."""
    lookup: Dict[str, int] = {}

    def normalized_variants(label: str) -> Set[str]:
        base = label.strip().lower()
        if not base:
            return set()
        variants = {
            base,
            base.replace(" ", ""),
            base.replace(" ", "_"),
            base.replace("_", " "),
            base.replace("_", ""),
            base.replace("-", " "),
            base.replace("-", ""),
        }
        return {v for v in variants if v}

    for idx, name in model_names.items():
        for variant in normalized_variants(name):
            lookup[variant] = idx

        synonyms = CLASS_SYNONYMS.get(name.lower(), [])
        for synonym in synonyms:
            for variant in normalized_variants(synonym):
                lookup.setdefault(variant, idx)

    return lookup


def resolve_targets(words: Sequence[str], model_names: Dict[int, str]) -> Tuple[Set[int], List[str]]:
    """Map user-provided words to model class ids."""
    lookup = _build_label_lookup(model_names)

    resolved: Set[int] = set()
    unmatched: List[str] = []
    for word in words:
        base = word.strip().lower()
        variants = {
            base,
            base.replace(" ", ""),
            base.replace(" ", "_"),
            base.replace("_", " "),
            base.replace("_", ""),
            base.replace("-", " "),
            base.replace("-", ""),
        }
        found = None
        for candidate in variants:
            if candidate in lookup:
                found = lookup[candidate]
                break

        if found is None:
            unmatched.append(word.strip())
        else:
            resolved.add(found)

    return resolved, unmatched


def interactive_target_selection(
    known_names: Sequence[str],
    model_names: Dict[int, str],
    initial_words: Sequence[str] | None = None,
) -> Tuple[Set[int] | None, List[str]]:
    """Loop until the user provides at least one valid target or exits."""
    pending_words = [w for w in initial_words] if initial_words else None

    while True:
        if pending_words is not None:
            words = pending_words
            pending_words = None
        else:
            words = prompt_for_targets()

        if not words:
            print("Veuillez entrer au moins un mot-clé.")
            continue

        lowered = {w.lower() for w in words}
        if any(token in {"exit", "quit"} for token in lowered):
            return None, []

        target_ids, unmatched = resolve_targets(words, model_names)
        if unmatched:
            print("Mots non reconnus :", ", ".join(unmatched))

        if not target_ids:
            print("Aucune classe valide trouvée, réessayez.")
            continue

        return target_ids, unmatched


def draw_targets(
    result,
    class_names: Dict[int, str],
    min_conf: float,
    requested_ids: Set[int],
    color_map: Dict[int, Tuple[int, int, int]] | None = None,
    show_missing_text: bool = True,
):
    """Render requested detections with per-class colors and optional missing notice."""
    frame = result.orig_img.copy()
    boxes = result.boxes

    present_ids: Set[int] = set()
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        present_ids = set()
    else:
        cls = boxes.cls.cpu().numpy()
        raw_conf = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(cls)

        for idx in range(len(cls)):
            if float(raw_conf[idx]) >= min_conf:
                present_ids.add(int(cls[idx]))

        # Draw kept boxes
        for idx in range(len(cls)):
            if float(raw_conf[idx]) < min_conf:
                continue
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[idx].tolist()]
            cid = int(cls[idx])
            label = f"{class_names[cid]} {float(raw_conf[idx]):.2f}"
            color = (0, 255, 0)
            if color_map and cid in color_map:
                color = color_map[cid]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Larger label text for better readability
            font_scale = 1.0
            text_thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            label_y = y1 - 8
            if label_y < label_h:
                label_y = y1 + label_h + 8
            bg_top = max(label_y - label_h - baseline - 4, 0)
            bg_bottom = min(label_y + baseline, frame.shape[0])
            bg_right = min(x1 + label_w + 8, frame.shape[1])
            cv2.rectangle(frame, (x1, bg_top), (bg_right, bg_bottom), color, cv2.FILLED)
            text_origin = (x1 + 4, min(label_y, frame.shape[0] - baseline - 2))
            cv2.putText(frame, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

    # Missing notice
    if show_missing_text and requested_ids:
        missing_ids = [cid for cid in requested_ids if cid not in present_ids]
        if missing_ids:
            missing_labels = ", ".join(class_names[cid] for cid in missing_ids)
            cv2.putText(
                frame,
                f"Non detecte: {missing_labels}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    # Help overlay
    help_text = "Tapez des objets dans le terminal | q/Echap: quitter"
    (tw, th), bl = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(th + 8, frame.shape[0] - 10)
    x = 16
    cv2.putText(frame, help_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def run_on_image(
    model,
    path: str,
    window_name: str,
    screen_w: int,
    screen_h: int,
    state: 'TargetState',
    imgsz: int,
    min_conf: float,
    stop_event: 'threading.Event',
) -> None:
    """Affiche une image et met à jour les détections à la volée selon l'entrée console."""
    orig = cv2.imread(path)
    if orig is None:
        print("Erreur : impossible de charger l'image.")
        return

    last_ids: List[int] = []
    annotated = orig.copy()
    while not stop_event.is_set():
        with state.lock:
            ids = list(state.ordered_ids)
            updated = state.updated
            state.updated = False

        if ids != last_ids or updated:
            if ids:
                results = model.predict(
                    source=path,
                    imgsz=imgsz,
                    conf=min_conf,
                    save=False,
                    stream=False,
                    verbose=False,
                    classes=sorted(set(ids)),
                )
                r = results[0] if isinstance(results, list) else results
                color_map = build_color_map(ids)
                annotated = draw_targets(r, r.names, min_conf, set(ids), color_map, show_missing_text=True)
            else:
                annotated = orig.copy()
                cv2.putText(
                    annotated,
                    "Tapez des objets dans le terminal (ex: avion & stylo)",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            last_ids = ids

        show_frame(window_name, annotated, screen_w, screen_h)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            stop_event.set()
            break


def run_on_video(
    model,
    path: str,
    window_name: str,
    screen_w: int,
    screen_h: int,
    state: 'TargetState',
    imgsz: int,
    min_conf: float,
    stop_event: 'threading.Event',
) -> None:
    """Lit une vidéo en boucle et met à jour les détections selon l'entrée console."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la video.")
        return
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            with state.lock:
                ids = list(state.ordered_ids)
            if ids:
                results = model.predict(
                    source=frame,
                    imgsz=imgsz,
                    conf=min_conf,
                    save=False,
                    stream=False,
                    verbose=False,
                    classes=sorted(set(ids)),
                )
                r = results[0] if isinstance(results, list) else results
                color_map = build_color_map(ids)
                annotated = draw_targets(r, r.names, min_conf, set(ids), color_map, show_missing_text=True)
            else:
                annotated = frame
                cv2.putText(
                    annotated,
                    "Tapez des objets dans le terminal (ex: avion & stylo)",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            show_frame(window_name, annotated, screen_w, screen_h)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                stop_event.set()
                break
    finally:
        cap.release()


def pick_source(path: str) -> str:
    """Retourne un fichier image/vidéo valide.

    - Si `path` est un fichier image/vidéo: le retourne.
    - Si `path` est un dossier: choisit le premier média valide trouvé.
    - Sinon: lève une ValueError.
    """
    if os.path.isfile(path):
        if is_image(path) or is_video(path):
            return path
        raise ValueError(f"Format de fichier non supporte: {path}")

    if os.path.isdir(path):
        entries = sorted(os.listdir(path))
        for name in entries:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate) and (is_image(candidate) or is_video(candidate)):
                return candidate
        raise ValueError("Aucun fichier image ou video valide trouve dans le dossier.")

    raise ValueError(f"Chemin invalide: {path}")


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur CLI pour ce script."""
    parser = argparse.ArgumentParser(description="Detection filtree par mots-cles pour YOLOv8.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Chemin vers le poids YOLO a charger.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Image, video ou dossier a analyser.")
    parser.add_argument("--targets", nargs="*", help="Liste de mots-cles cibles (supporte '&').")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'entree pour le modele.")
    parser.add_argument("--conf", type=float, default=MIN_CONFIDENCE, help="Seuil de confiance minimal (>= 0.35).")
    return parser


def main():
    """Point d'entrée: configure le modèle, lance la fenêtre et la saisie continue."""
    parser = build_parser()
    args = parser.parse_args()

    min_conf = max(args.conf, MIN_CONFIDENCE)
    if args.conf < MIN_CONFIDENCE:
        print(f"Seuil de confiance relevé à {MIN_CONFIDENCE:.2f} pour respecter la limite fixée.")

    model = YOLO(args.model)
    known_names = list_known_classes(model.names)

    try:
        selected = pick_source(args.source)
    except ValueError as error:
        print(f"Erreur : {error}")
        return

    screen_w, screen_h = get_screen_size()
    preview_window = "Apercu source"

    print("\nUtilisation :")
    print(" - Separez plusieurs objets avec '&' (ex: avion & stylo)")
    print(" - Tapez 'exit' pour quitter")
    print(" - L'image reste affichee; tapez des objets dans la console. 'q'/Echap pour quitter.\n")

    print_known_classes(known_names)

    window_name = "Detections cibles"
    # Shared state + input thread
    state = TargetState()
    if args.targets:
        ordered_ids, unmatched = resolve_ordered(args.targets, model.names)
        with state.lock:
            state.ordered_ids = ordered_ids
            state.unmatched = unmatched
            state.updated = True
        if unmatched:
            print("Mots non reconnus :", ", ".join(unmatched))
        if ordered_ids:
            print("Recherche des classes :", ", ".join(model.names[i] for i in ordered_ids))

    stop_event = threading.Event()
    th = threading.Thread(target=input_loop, args=(state, model.names, stop_event), daemon=True)
    th.start()

    if is_image(selected):
        run_on_image(model, selected, window_name, screen_w, screen_h, state, args.imgsz, min_conf, stop_event)
    else:
        run_on_video(model, selected, window_name, screen_w, screen_h, state, args.imgsz, min_conf, stop_event)

    cv2.destroyAllWindows()
    return



if __name__ == "__main__":
    main()
