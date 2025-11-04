from pathlib import Path
import random
import shutil

# Répertoires
ROOT = Path(".")
IMG_TRAIN = ROOT / "images" / "train"
LBL_TRAIN = ROOT / "labels" / "train"
IMG_VAL   = ROOT / "images" / "val"
LBL_VAL   = ROOT / "labels" / "val"

# Paramètres
VAL_RATIO = 0.2   # 20% pour validation
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def main():
    assert IMG_TRAIN.exists() and LBL_TRAIN.exists(), "images/train ou labels/train manquant."
    IMG_VAL.mkdir(parents=True, exist_ok=True)
    LBL_VAL.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in IMG_TRAIN.iterdir() if p.suffix.lower() in IMG_EXTS]
    imgs = sorted(imgs)
    print(f"Images train trouvées: {len(imgs)}")

    # On ne déplace que celles qui ont un .txt correspondant
    pairs = []
    for img in imgs:
        lbl = (LBL_TRAIN / img.stem).with_suffix(".txt")
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            print(f"⚠️ Pas de label pour {img.name} (ignoré)")

    random.seed(SEED)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_RATIO))
    val_pairs = pairs[:n_val]

    print(f"Déplacement vers val: {len(val_pairs)} paires")

    for img, lbl in val_pairs:
        shutil.move(str(img), str(IMG_VAL / img.name))
        shutil.move(str(lbl), str(LBL_VAL / lbl.name))

    print("✅ Split terminé.")
    print(f"Train: {len(list(IMG_TRAIN.glob('*')))} images | Val: {len(list(IMG_VAL.glob('*')))} images")

    # ⚠️ Supprime les caches YOLO pour qu'il rescane correctement
    for cache in [LBL_TRAIN.with_suffix(".cache"), LBL_VAL.with_suffix(".cache")]:
        if cache.exists():
            cache.unlink()
            print(f"🗑️ Cache supprimé: {cache}")

if __name__ == "__main__":
    main()
