# -*- coding: utf-8 -*-
"""
Édite les .txt (format YOLO) et renomme les paires .txt + image.
Flux :
  0) Supprime les images orphelines (sans .txt associé, même stem)
  1) (optionnel) Modifie le premier identifiant de classe dans les .txt
  2) Renomme les paires .txt + image avec sécurité (deux passes)

CONFIGURE ci-dessous puis exécute (Run ▶) dans VS Code.
"""

from pathlib import Path
import os
import re
import sys
import uuid
from typing import Dict, List, Optional, Tuple

# ===================== CONFIG =====================
FOLDER      = r".\videos\images_de_videos"       # Dossier à traiter
FILE_PREFIX = "img_avion"           # Filtre: ne prendre que les fichiers dont le nom commence par ce préfixe (laisser "" pour tout)
RECURSIVE   = False                   # True: parcourir les sous-dossiers ; False: dossier courant uniquement

# --- Édition des classes dans les .txt ---
ENABLE_CLASS_EDIT = True              # Activer le remplacement du premier identifiant de classe
CLASS_FROM = 16                       # Remplacer si le premier token == CLASS_FROM ...
CLASS_TO   = 0                        # ... par CLASS_TO
BACKUP     = True                     # Créer un .bak du .txt avant écriture

# --- Renommage des fichiers ---
ENABLE_RENAME = True                  # Activer le renommage .txt + image
RENAME_REQUIRE_PAIR = True            # True: ne renommer que si .txt ET image existent (recommandé)
IMAGE_EXTS = ("jpg", "jpeg", "png")   # Extensions d'image possibles (ordre de préférence à l'association)

# Patron du nouveau nom (sans extension) via str.format :
# Variables disponibles : NEW_PREFIX, seq, old_num, old_stem, i (index base 0)
# Exemples :
#   "{NEW_PREFIX}_{seq}"        -> img_ciseaux_216
#   "{NEW_PREFIX}_{seq:03d}"    -> img_ciseaux_216 (avec padding si seq=216 -> "216")
#   "{old_stem}_v2"             -> garde l'ancien nom et ajoute suffixe
NAME_PATTERN  = "{NEW_PREFIX}_{seq}"
NEW_PREFIX    = "img_avion"         # Utilisé par NAME_PATTERN
RENAME_START_AT = 1              # Numéro de départ pour 'seq' (numéro séquentiel des nouveaux noms)

# --- Nettoyage orphelins ---
DELETE_ORPHAN_IMAGES = True           # True: supprimer les images sans .txt correspondant (même stem)

# --- Exécution ---
DRY_RUN = False                        # True: simulation (affiche le plan), False: applique
FORCE   = True                     # True: autoriser l'écrasement si une destination existe déjà (déconseillé)
# ==================================================

# Regex pour identifier un numéro final dans un nom (ex: "img_ciseaux_123" -> 123)
RX_TRAILING_NUM = re.compile(r".*?(\d+)$")

def iter_txt_files(root: Path) -> List[Path]:
    files = list(root.rglob("*.txt")) if RECURSIVE else list(root.glob("*.txt"))
    if FILE_PREFIX:
        files = [p for p in files if p.stem.startswith(FILE_PREFIX)]
    return sorted(files, key=lambda p: p.name.lower())

def iter_image_files(root: Path) -> List[Path]:
    pats = [f"*.{ext}" for ext in IMAGE_EXTS]
    files: List[Path] = []
    for pat in pats:
        files.extend(root.rglob(pat) if RECURSIVE else root.glob(pat))
    if FILE_PREFIX:
        files = [p for p in files if p.stem.startswith(FILE_PREFIX)]
    return sorted(files, key=lambda p: p.name.lower())

def find_image_for_stem(folder: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        candidate = folder / f"{stem}.{ext}"
        if candidate.exists():
            return candidate
    return None

def parse_old_num_from_stem(stem: str) -> Optional[int]:
    m = RX_TRAILING_NUM.match(stem)
    return int(m.group(1)) if m else None

def compile_class_regex() -> re.Pattern:
    # Ex: pour CLASS_FROM=0 -> r'^(\s*)0(\s+)' (en MULTILINE)
    return re.compile(rf'^(\s*){re.escape(str(CLASS_FROM))}(\s+)', re.MULTILINE)

def transform_txt_content(text: str, rx_class: re.Pattern) -> str:
    # Remplace UNIQUEMENT le premier token égal à CLASS_FROM, en conservant les espaces
    # IMPORTANT: utiliser \g<1> et \g<2> (évite l'ambiguïté \11)
    return rx_class.sub(r'\g<1>' + str(CLASS_TO) + r'\g<2>', text)

def safe_read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def safe_write_text(p: Path, content: str):
    p.write_text(content, encoding="utf-8")

def build_new_stem(old_stem: str, old_num: Optional[int], seq: int, i: int) -> str:
    return NAME_PATTERN.format(
        NEW_PREFIX=NEW_PREFIX,
        seq=seq,
        old_num=old_num if old_num is not None else "",
        old_stem=old_stem,
        i=i
    )

def detect_external_conflicts(mappings: List[Tuple[Path, Path]]) -> List[Path]:
    """Retourne les destinations qui existent déjà et ne sont pas des sources (risque d'écrasement)."""
    srcs = {src.resolve() for src, _ in mappings}
    conflicts = []
    for _, dst in mappings:
        if dst.exists() and dst.resolve() not in srcs:
            conflicts.append(dst)
    return conflicts

def two_pass_move(mappings: List[Tuple[Path, Path]]):
    """Renomme en deux passes (src -> tmp -> dst) pour éviter les collisions entre sources."""
    if not mappings:
        return
    tmp_map: Dict[Path, Path] = {}
    for src, _ in mappings:
        tmp = src.parent / f"__tmp__{uuid.uuid4().hex}__{src.name}"
        tmp_map[src] = tmp

    # 1) src -> tmp
    for src, _ in mappings:
        tmp = tmp_map[src]
        if DRY_RUN:
            print(f"[DRY] {src.name}  ->  {tmp.name}")
        else:
            os.replace(src, tmp)

    # 2) tmp -> dst
    for src, dst in mappings:
        tmp = tmp_map[src]
        if DRY_RUN:
            print(f"[DRY] {tmp.name}  ->  {dst.name}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if FORCE:
                    os.remove(dst)
                else:
                    raise FileExistsError(f"Destination existe déjà: {dst}")
            os.replace(tmp, dst)

def delete_orphan_images(root: Path, txt_stems: set) -> int:
    """
    Supprime les images dont le stem n'a pas de .txt correspondant.
    Respecte FILE_PREFIX et RECURSIVE. Affiche ce qui est fait en DRY_RUN.
    Retourne le nombre d'images supprimées (ou à supprimer en DRY).
    """
    images = iter_image_files(root)
    to_delete: List[Path] = []
    for img in images:
        # .txt attendu dans le même dossier et même stem
        expected_txt = img.with_suffix(".txt")
        if FILE_PREFIX and not img.stem.startswith(FILE_PREFIX):
            continue
        if not expected_txt.exists():
            to_delete.append(img)

    if not to_delete:
        print("Aucune image orpheline détectée.")
        return 0

    print(f"\nImages orphelines détectées (sans .txt): {len(to_delete)}")
    for p in to_delete:
        if DRY_RUN:
            print(f"[DRY][DEL] {p}")
        else:
            try:
                p.unlink()
                print(f"[DEL] {p}")
            except Exception as e:
                print(f"[ERR] Échec suppression {p}: {e}")
    return len(to_delete)

def main():
    root = Path(FOLDER).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Erreur: dossier introuvable: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Dossier: {root}")
    print(f"Mode: {'DRY-RUN' if DRY_RUN else 'EXÉCUTION'} — BACKUP={BACKUP} — "
          f"CLASS_EDIT={'ON' if ENABLE_CLASS_EDIT else 'OFF'} — RENAME={'ON' if ENABLE_RENAME else 'OFF'} — "
          f"DELETE_ORPHAN_IMAGES={'ON' if DELETE_ORPHAN_IMAGES else 'OFF'}")

    # 0) Nettoyage des images orphelines AVANT TOUT
    if DELETE_ORPHAN_IMAGES:
        # info: on ne se base pas sur txt_stems ici, on teste l'existence d'un .txt à côté de chaque image
        delete_orphan_images(root, txt_stems=set())

    # (Re)lister les .txt après nettoyage
    txt_files = iter_txt_files(root)
    if not txt_files:
        print("Aucun .txt correspondant après nettoyage.")
        if DRY_RUN:
            print("[DRY-RUN] Aucune écriture finale. Mets DRY_RUN=False pour appliquer.")
        return

    print(f"Fichiers .txt trouvés: {len(txt_files)} (prefix='{FILE_PREFIX or '*'}', recursive={RECURSIVE})")

    # 1) Édition du contenu des .txt (avant renommage)
    if ENABLE_CLASS_EDIT:
        rx_class = compile_class_regex()
        for p in txt_files:
            original = safe_read_text(p)
            transformed = transform_txt_content(original, rx_class)
            if original != transformed:
                if DRY_RUN:
                    print(f"* {p.name} : CHANGEMENTS PRÉVUS (classe {CLASS_FROM} -> {CLASS_TO})")
                else:
                    if BACKUP:
                        bak = p.with_suffix(p.suffix + ".bak")
                        safe_write_text(bak, original)
                    safe_write_text(p, transformed)
                    print(f"✓ {p.name} : contenu modifié" + (" (+ .bak)" if BACKUP else ""))
            else:
                print(f"= {p.name} : aucun changement de classe")

    if ENABLE_RENAME:
        # 2) Préparation du plan de renommage pour les paires
        entries: List[Tuple[Path, Optional[Path], Optional[int]]] = []
        for txt in txt_files:
            stem = txt.stem
            img = find_image_for_stem(txt.parent, stem)
            if RENAME_REQUIRE_PAIR and img is None:
                # ignorer si pas d'image associée
                continue
            entries.append((txt, img, parse_old_num_from_stem(stem)))

        if not entries:
            print("\nAucune entrée à renommer (conditions non remplies, voir RENAME_REQUIRE_PAIR).")
            if DRY_RUN:
                print("[DRY-RUN] Aucune écriture finale. Mets DRY_RUN=False pour appliquer.")
            return

        # Tri par ancien numéro si disponible, sinon par nom
        def sort_key(tup):
            txt, _, old_num = tup
            return (old_num is None, old_num if old_num is not None else 0, txt.name.lower())

        entries.sort(key=sort_key)

        mappings: List[Tuple[Path, Path]] = []
        seq = RENAME_START_AT
        print("\nPlan de renommage (paires sélectionnées):")
        for i, (txt, img, old_num) in enumerate(entries):
            old_stem = txt.stem
            new_stem = build_new_stem(old_stem, old_num, seq, i)
            seq += 1

            # .txt
            new_txt = txt.with_name(new_stem + ".txt")
            if new_txt != txt:
                mappings.append((txt, new_txt))
                print(f" - {txt.name}  ->  {new_txt.name}")

            # image (si présente)
            if img is not None:
                new_img = img.with_name(new_stem + img.suffix.lower())
                if new_img != img:
                    mappings.append((img, new_img))
                    print(f" - {img.name}  ->  {new_img.name}")

        # 3) Détection conflits externes
        conflicts = detect_external_conflicts(mappings)
        if conflicts and not FORCE:
            print("\nABANDON: des destinations existent déjà (risque d'écrasement).")
            for c in conflicts:
                print(f" - {c}")
            print("→ Si c'est attendu, mets FORCE=True puis relance.")
            sys.exit(2)

        # 4) Renommage en deux passes
        print("\nRenommage (deux passes):")
        two_pass_move(mappings)

    if DRY_RUN:
        print("\n[DRY-RUN] Aucune écriture finale. Mets DRY_RUN=False pour appliquer.")

if __name__ == "__main__":
    main()
