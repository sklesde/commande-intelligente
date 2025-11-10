import cv2
from pathlib import Path

# --- Paramètres à personnaliser ---
video_path = r".\videos\Avion4.mov"
output_dir = Path("./") / "videos/images_de_videos"
name = "ciseaux"          # 🔧 <-- choisis ici ton nom (ex: "test", "capture", etc.)
start_index_default = 0   # nombre de départ si aucun fichier n’existe
X = 5                     # 1 image sur X

# --- Préparation du dossier ---
output_dir.mkdir(parents=True, exist_ok=True)

# --- Trouver le dernier index existant ---
pattern = f"img_{name}_*.jpg"
existing_files = sorted(output_dir.glob(pattern))

if existing_files:
    # Ex: img_ciseaux_123.jpg -> récupère 123
    last_num = max(int(f.stem.split('_')[-1]) for f in existing_files)
    start_index = last_num + 1
else:
    start_index = start_index_default

print(f"➡️  Démarrage à l’image n°{start_index} (pas d’écrasement des anciennes)")

# --- Ouvrir la vidéo ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erreur : impossible d’ouvrir la vidéo :", video_path)
    raise SystemExit(1)

# --- Lecture de la vidéo ---
frame_idx = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % X == 0:
        # Format sur 3 chiffres : 001, 002, 003...
        filename = output_dir / f"img_{name}_{start_index + saved_count:03d}.jpg"
        cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_count += 1

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

print(f"✅ Terminé : {saved_count} nouvelle(s) image(s) enregistrée(s) dans {output_dir.resolve()}")
