from pathlib import Path

# === Paramètres ===
# Racine du projet (là où se trouvent images/ et labels/)
ROOT = Path(".")

# Traitement de ces splits (met "val" si tu veux aussi le corriger)
SPLITS = ["train"]  # ou ["train", "val"]

# Correspondance "mot-clé dans le nom du fichier" -> ID YOLO
# (insensible à la casse)
KEYWORD_TO_ID = {
    "avion": 0,     # Plane
    "ciseaux": 1,   # Cisors
    "colle": 2,     # Glue
    "stylo": 3,     # Pen
}

# Sauvegarder un .bak avant modification ?
MAKE_BACKUP = True

# === Code ===
def detect_class_id_from_name(stem: str):
    s = stem.lower()
    for kw, cid in KEYWORD_TO_ID.items():
        if kw in s:
            return cid, kw
    return None, None

def fix_label_file(txt_path: Path, new_id: int):
    text = txt_path.read_text(encoding="utf-8").strip()
    if not text:
        return 0, 0  # fichier vide

    lines = text.splitlines()
    changed = 0
    kept = 0
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            new_lines.append(line)
            continue
        # Remplacer l'ID de classe (colonne 0)
        old_id = parts[0]
        if old_id != str(new_id):
            parts[0] = str(new_id)
            changed += 1
        else:
            kept += 1
        new_lines.append(" ".join(parts))

    if MAKE_BACKUP:
        bak = txt_path.with_suffix(txt_path.suffix + ".bak")
        bak.write_text(text, encoding="utf-8")

    txt_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return changed, kept

def main():
    total_files = 0
    total_changed_lines = 0
    total_kept_lines = 0
    unknown_files = []

    for split in SPLITS:
        lbl_dir = ROOT / "labels" / split
        if not lbl_dir.exists():
            print(f"[{split}] Dossier inexistant : {lbl_dir} (on passe)")
            continue

        txt_files = sorted(lbl_dir.glob("*.txt"))
        print(f"\n=== Split: {split} | {len(txt_files)} fichiers .txt ===")

        for txt in txt_files:
            cid, kw = detect_class_id_from_name(txt.stem)
            if cid is None:
                unknown_files.append(txt)
                print(f"  ⚠️  Classe non détectée depuis le nom : {txt.name}")
                continue

            changed, kept = fix_label_file(txt, cid)
            total_files += 1
            total_changed_lines += changed
            total_kept_lines += kept

            if changed > 0:
                print(f"  ✅ {txt.name} → classe '{kw}' (ID {cid}) | {changed} ligne(s) corrigée(s)")
            else:
                print(f"  ✓ {txt.name} déjà correct (ID {cid})")

    print("\n=== RÉSUMÉ ===")
    print(f"Fichiers traités     : {total_files}")
    print(f"Lignes modifiées     : {total_changed_lines}")
    print(f"Lignes déjà correctes: {total_kept_lines}")
    if unknown_files:
        print(f"\n⚠️ Fichiers ignorés (aucun mot-clé trouvé dans le nom) : {len(unknown_files)}")
        for f in unknown_files[:10]:
            print("   -", f)

if __name__ == "__main__":
    main()
