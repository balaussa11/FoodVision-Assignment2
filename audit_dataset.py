"""Check archive-derived folder structure, file readability and exact leakage."""
import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='food_dataset')
    args = p.parse_args()
    root = Path(args.data_dir)
    groups = Counter()
    digests = defaultdict(list)
    broken = []
    for split in ('train', 'val'):
        for cls in ('bread', 'drinks', 'fruit'):
            for f in (root / split / cls).glob('*'):
                if not f.is_file() or f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue
                groups[f'{split}/{cls}'] += 1
                try:
                    with Image.open(f) as im:
                        im.verify()
                    digests[hashlib.sha256(f.read_bytes()).hexdigest()].append(str(f))
                except Exception as err:
                    broken.append((str(f), str(err)))
    print('Counts:', dict(sorted(groups.items())))
    print('Broken files:', broken)
    duplicates = [p for p in digests.values() if len(p) > 1]
    print('Exact duplicate groups:', duplicates)
    if broken:
        raise SystemExit('Please fix unreadable images before training.')


if __name__ == '__main__':
    main()
