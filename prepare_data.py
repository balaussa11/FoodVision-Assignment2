"""Extract the exact A1 dataset ZIP safely into the project folder."""
import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', dest='zip_path', default='food_dataset.zip')
    parser.add_argument('--out', default='.')
    args = parser.parse_args()
    archive = Path(args.zip_path)
    if not archive.exists():
        raise FileNotFoundError(f'Cannot find {archive}. Put your A1 ZIP beside this script.')
    destination = Path(args.out).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        invalid = z.testzip()
        if invalid:
            raise ValueError(f'Bad archive member: {invalid}')
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'Unsafe ZIP path: {info.filename}')
        z.extractall(destination)
    for split in ('train', 'val'):
        for cls in ('bread', 'drinks', 'fruit'):
            folder = destination / 'food_dataset' / split / cls
            if not folder.exists():
                raise ValueError(f'Expected folder missing: {folder}')
            count = len([p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            print(f'{split}/{cls}: {count} images')
    print('Dataset extracted successfully.')


if __name__ == '__main__':
    main()
