from ultralytics import YOLO
import os
import cv2
import math

# ==========================
# Réglages utilisateur
# ==========================
MODEL_PATH = "runs_plane/yolov8n_plane/weights/best.pt"
SOURCE = "test/video_test_2.mp4"  # ➤ mets ici ton fichier (image/vidéo) ou un dossier
CONF_THRESHOLD = 0.35  # Masque les détections en dessous de 35 %

# ==========================
# Utilitaires
# ==========================
IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".m4v"]

def is_image(path: str) -> bool:
    return any(path.lower().endswith(ext) for ext in IMAGE_EXT)

def is_video(path: str) -> bool:
    return any(path.lower().endswith(ext) for ext in VIDEO_EXT)

def get_screen_size():
    """
    Récupère la taille de l'écran. Si indisponible, fallback 1920x1080.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080

def resize_to_fit(img, max_w, max_h, add_letterbox=True):
    """
    Redimensionne l'image pour tenir dans (max_w, max_h) en conservant le ratio.
    Letterbox (bandes noires) optionnel pour occuper exactement la zone.
    """
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return img

    scale = min(max_w / w, max_h / h)
    # éviter d'agrandir trop si inutile : on accepte l'agrandissement contrôlé
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    if not add_letterbox:
        return resized

    # Crée un canvas noir de la taille cible et centre l'image redimensionnée
    canvas = (0, 0, 0)
    out = cv2.cvtColor(cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA), cv2.COLOR_BGRA2BGR)  # safe copy
    boxed = img  # placeholder just to keep names
    boxed = (255 * 0)  # dummy to avoid lints (not used)

    # Pour garder le cadre <= max_w x max_h mais proprement visible
    canvas_img = None
    if new_w == max_w and new_h == max_h:
        return resized  # remplit exactement

    canvas_img = (cv2.UMat if hasattr(cv2, 'UMat') else None)
    # Crée l'image de fond noire
    frame = cv2.copyMakeBorder(
        resized,
        top=(max_h - new_h)//2,
        bottom=math.ceil((max_h - new_h)/2),
        left=(max_w - new_w)//2,
        right=math.ceil((max_w - new_w)/2),
        borderType=cv2.BORDER_CONSTANT,
        value=canvas
    )
    return frame

def best_display_size(screen_w, screen_h, margin_ratio=0.9):
    """
    Réserve une marge (ex: 90%) de l'écran pour éviter les dépassements (barres d'OS, etc.).
    """
    return int(screen_w * margin_ratio), int(screen_h * margin_ratio)

def show_frame(window_name, frame, screen_w, screen_h):
    max_w, max_h = best_display_size(screen_w, screen_h, margin_ratio=0.9)
    display = resize_to_fit(frame, max_w, max_h, add_letterbox=True)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Ajuste la fenêtre à la taille du frame affiché
    h, w = display.shape[:2]
    cv2.resizeWindow(window_name, w, h)
    cv2.imshow(window_name, display)

# ==========================
# Inference + Affichage
# ==========================
def run_on_image(model, path, window_name, screen_w, screen_h):
    results = model.predict(
        source=path,
        imgsz=640,
        conf=CONF_THRESHOLD,
        save=False,
        stream=False,
        verbose=False,
    )
    # results peut être une liste (une image) → on itère par sécurité
    for r in (results if isinstance(results, list) else [results]):
        annotated = r.plot()  # BGR
        show_frame(window_name, annotated, screen_w, screen_h)
        # Attente jusqu'à fermeture par l'utilisateur
        print("Appuie sur 'q' ou 'Échap' pour fermer.")
        while True:
            key = cv2.waitKey(16) & 0xFF
            if key in (27, ord('q')):  # ESC ou q
                break
        cv2.destroyWindow(window_name)

def run_on_video(model, path, window_name, screen_w, screen_h):
    # Utilise le flux de prédictions pour traiter frame par frame
    for r in model.predict(
        source=path,
        imgsz=640,
        conf=CONF_THRESHOLD,
        save=False,
        stream=True,
        verbose=False,
    ):
        annotated = r.plot()  # BGR annoté
        show_frame(window_name, annotated, screen_w, screen_h)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC ou q pour quitter
            break
        # Space pour pause
        if key == 32:  # SPACE
            # Pause jusqu'à nouvelle pression d'espace/q/esc
            while True:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 in (27, ord('q'), 32):
                    if key2 in (27, ord('q')):
                        cv2.destroyWindow(window_name)
                        return
                    else:
                        break
    cv2.destroyWindow(window_name)

def main():
    # Charge modèle
    model = YOLO(MODEL_PATH)

    # Détecte résolution d'écran (pour garantir affichage complet)
    screen_w, screen_h = get_screen_size()

    # Sélectionne une seule entrée image/vidéo
    selected = None
    if os.path.isfile(SOURCE):
        if is_image(SOURCE) or is_video(SOURCE):
            selected = SOURCE
        else:
            print("❌ Fichier non supporté :", SOURCE)
            return
    elif os.path.isdir(SOURCE):
        # Ne prend qu'un seul premier fichier image/vidéo pour respecter la consigne
        entries = sorted(os.listdir(SOURCE))
        for f in entries:
            p = os.path.join(SOURCE, f)
            if os.path.isfile(p) and (is_image(p) or is_video(p)):
                selected = p
                break
        if selected is None:
            print("❌ Aucun fichier image/vidéo valide trouvé dans le dossier.")
            return
    else:
        print("❌ Chemin invalide :", SOURCE)
        return

    window_name = "Détections (ratio conservé • full visible)"

    if is_image(selected):
        print(f"📷 Image détectée : {selected}")
        run_on_image(model, selected, window_name, screen_w, screen_h)
    elif is_video(selected):
        print(f"🎥 Vidéo détectée : {selected}")
        run_on_video(model, selected, window_name, screen_w, screen_h)

if __name__ == "__main__":
    main()
