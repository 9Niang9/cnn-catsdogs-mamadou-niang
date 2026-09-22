from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARCHIVE_DIR = ROOT / "downloads"
LOCAL_DATASET = ROOT.parent / "cnn_cat_dog_image_classification" / "Cat_Dog_data"


def has_expected_structure(folder: Path) -> bool:
    required = [
        folder / "train" / "cat",
        folder / "train" / "dog",
        folder / "test" / "cat",
        folder / "test" / "dog",
    ]
    return all(path.exists() for path in required)


def prepare_local_dataset() -> Path:
    if has_expected_structure(DATA_DIR):
        print(f"Jeux de données déjà disponible dans : {DATA_DIR}")
        return DATA_DIR

    if LOCAL_DATASET.exists():
        print(f"Utilisation du dataset local situé dans : {LOCAL_DATASET}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if DATA_DIR != LOCAL_DATASET:
            for item in LOCAL_DATASET.iterdir():
                target = DATA_DIR / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
        print(f"Jeu de données prêt dans : {DATA_DIR}")
        return DATA_DIR

    raise FileNotFoundError(
        "Aucun dataset local Cats vs Dogs trouvé. Attendu : "
        f"{LOCAL_DATASET}. Téléchargez le dataset Kaggle ou placez-le à cet emplacement."
    )


def prepare_kaggle_dataset() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "KaggleHub n’est pas installé. Installez-le avec : "
            "python -m pip install kagglehub"
        ) from exc

    print("Téléchargement de l’archive Cats vs Dogs depuis KaggleHub...")
    path = kagglehub.competition_download("dogs-vs-cats", force=False)
    archive_path = Path(path)

    if archive_path.is_dir():
        zip_files = list(archive_path.glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError(f"Aucun fichier zip trouvé dans {archive_path}")
        archive_path = zip_files[0]
    elif not archive_path.exists():
        raise FileNotFoundError(f"Archive introuvable : {archive_path}")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    target = ARCHIVE_DIR / archive_path.name
    if archive_path != target:
        shutil.copy2(archive_path, target)

    print("Extraction de l’archive...")
    with zipfile.ZipFile(target, "r") as zf:
        zf.extractall(DATA_DIR)

    print(f"Jeu de données prêt dans : {DATA_DIR}")
    return DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Prépare le dataset Cats vs Dogs pour l’entraînement.")
    parser.add_argument(
        "--use-kaggle",
        action="store_true",
        help="Télécharge le dataset via KaggleHub si le dataset local n’est pas disponible.",
    )
    args = parser.parse_args()

    try:
        if has_expected_structure(DATA_DIR):
            print(f"Jeux de données déjà disponible dans : {DATA_DIR}")
        elif LOCAL_DATASET.exists():
            prepare_local_dataset()
        elif args.use_kaggle:
            prepare_kaggle_dataset()
        else:
            raise FileNotFoundError(
                f"Aucun dataset trouvé dans {DATA_DIR} ou {LOCAL_DATASET}. "
                "Utilisez le dataset local déjà présent dans le workspace, ou lancez ce script avec --use-kaggle."
            )
    except FileNotFoundError as exc:
        print(f"[ERREUR] {exc}")
        raise

    print("Structure attendue :")
    print("data/train/cat ...")
    print("data/train/dog ...")
    print("data/test/cat ...")
    print("data/test/dog ...")


if __name__ == "__main__":
    main()
