# detect_webcam.py
# Détection temps réel avec la webcam (YOLOv8 + OpenCV)
# Usage:
#   python detect_webcam.py --weights runs_plane/yolov8n_plane/weights/best.pt
# Options utiles:
#   --camera 0          # index de la caméra (par défaut 0)
#   --imgsz 640         # taille d'entrée du modèle
#   --conf 0.35         # seuil de confiance minimum
#   --save out.mp4      # enregistre la vidéo annotée
#   --show              # force l'affichage de la fenêtre (par défaut oui)
#   --noshow            # n'affiche pas la fenêtre (utile si tu veux juste enregistrer)

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

MIN_CONFIDENCE = 0.35  # Mask detections below 35 %


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="runs_plane/yolov8n_plane/weights/best.pt",
                   help="Chemin des poids .pt")
    p.add_argument("--camera", type=int, default=0, help="Index caméra (0 = webcam intégrée)")
    p.add_argument("--imgsz", type=int, default=640, help="Taille d'image d'entrée du modèle")
    p.add_argument("--conf", type=float, default=MIN_CONFIDENCE, help="Seuil de confiance (>= 0.35)")
    p.add_argument("--save", type=str, default="", help="Fichier de sortie vidéo (ex: out.mp4). Vide = pas d'enregistrement")
    p.add_argument("--show", dest="show", action="store_true", help="Afficher la fenêtre")
    p.add_argument("--noshow", dest="show", action="store_false", help="Ne pas afficher la fenêtre")
    p.set_defaults(show=True)
    return p.parse_args()


def main():
    args = parse_args()
    conf_threshold = max(args.conf, MIN_CONFIDENCE)
    if args.conf < MIN_CONFIDENCE:
        print(f"[INFO] Seuil de confiance relevé à {MIN_CONFIDENCE:.2f} pour respecter la limite fixée.")

    # Sélection automatique du device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device = {device} (CUDA dispo: {torch.cuda.is_available()})")

    # Charge le modèle
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables: {weights_path}")

    model = YOLO(str(weights_path))

    # Ouvre la caméra
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra index {args.camera}")

    # Optionnel: fixe une résolution (décommente si besoin)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Enregistrement vidéo ?
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ou "XVID"
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 25  # fallback raisonnable
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        print(f"[INFO] Enregistrement dans: {args.save} ({w}x{h}@{fps:.1f}fps)")

    # Boucle de lecture
    prev_t = time.time()
    win_name = "YOLOv8 Webcam (q pour quitter)"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame non lue, on stoppe.")
                break

            # Inférence
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=conf_threshold,
                device=device,
                verbose=False
            )

            # Image annotée
            annotated = results[0].plot()

            # FPS overlay
            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Affichage
            if args.show:
                cv2.imshow(win_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Enregistrement
            if writer is not None:
                writer.write(annotated)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Fin propre.")

if __name__ == "__main__":
    main()
