from __future__ import annotations

import os
import zipfile
from pathlib import Path

import kagglehub


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
ARCHIVE_DIR = ROOT / 'downloads'


def main():
    print('Downloading Dogs vs Cats competition archive from KaggleHub...')
    path = kagglehub.competition_download('dogs-vs-cats', force=False)
    print('Archive downloaded to:', path)

    archive_path = Path(path)
    if archive_path.is_dir():
        zip_files = list(archive_path.glob('*.zip'))
        if not zip_files:
            raise FileNotFoundError(f'No zip file found in {archive_path}')
        archive_path = zip_files[0]
    elif not archive_path.exists():
        raise FileNotFoundError(f'Archive not found: {archive_path}')

    DATA_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)
    target = ARCHIVE_DIR / archive_path.name
    if archive_path != target:
        import shutil
        shutil.copy2(archive_path, target)

    print('Extracting archive...')
    with zipfile.ZipFile(target, 'r') as zf:
        zf.extractall(DATA_DIR)

    print(f'Dataset ready under: {DATA_DIR}')
    print('Expected structure:')
    print('data/train/cat ...')
    print('data/train/dog ...')
    print('data/test/cat ...')
    print('data/test/dog ...')


if __name__ == '__main__':
    main()
