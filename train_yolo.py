from ultralytics import YOLO

# modèle le plus léger : rapide même sans GPU
model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",      # fichier de configuration
    imgsz=480,             # taille des images (plus petit = plus rapide)
    epochs=30,             # nombre de passes sur le dataset
    batch=8,               # nombre d’images traitées en parallèle
    workers=0,             # pour Windows + CPU
    device='cpu',          # tu n’as pas de GPU CUDA
    project="runs_plane",  # dossier de sortie
    name="yolov8n_plane",  # nom du sous-dossier de run
    exist_ok=True
)
