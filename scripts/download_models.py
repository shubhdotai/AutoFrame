#!/usr/bin/env python3
"""Download only requested weights, verifying SHA-256 before installation."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--include-yolo', action='store_true')
    p.add_argument('--include-talkset', action='store_true')
    p.add_argument('--output', type=Path, default=Path('models'))
    args = p.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'models/manifest.json').read_text())
    names = ['pretrain_AVA.model']
    if args.include_talkset:
        names.append('finetuning_TalkSet.model')
    if args.include_yolo:
        names.append('yolov8x_person_face.pt')
    args.output.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = manifest[name]
        target = args.output / name
        if target.exists():
            if digest(target) != spec['sha256']:
                raise SystemExit(f'{target}: checksum mismatch; move it aside before retrying')
            print(f'OK {target}')
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=args.output, delete=False) as out:
                temporary = Path(out.name)
                print(f'Downloading {name}')
                with urllib.request.urlopen(spec['url'], timeout=60) as response:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
            if digest(temporary) != spec['sha256']:
                raise ValueError(f'{name}: downloaded checksum differs from manifest')
            temporary.replace(target)
            print(f'OK {target}')
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
