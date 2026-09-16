#!/usr/bin/env python3
"""
Download AutoClip model weights, verifying SHA-256 before installation.

Assets are grouped in ``models/manifest.json``. By default only the LR-ASD
checkpoint is fetched, which is all the Torch pipeline needs alongside the
macOS Vision face detector:

    python scripts/download_models.py                      # LR-ASD only (3.4 MB)
    python scripts/download_models.py --include-yolo       # + YOLO detector
    python scripts/download_models.py --all                # everything

Files come from the Hugging Face repo named by ``--repo`` (or the
``AUTOCLIP_MODELS_REPO`` environment variable). ``--source upstream`` fetches
from each artifact's original home instead, for anything the manifest records
an origin URL for.

No third-party dependency is required; this uses urllib and hashlib only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

# Mirror holding every file in models/ that AutoClip actually uses.
# Override with --repo or the AUTOCLIP_MODELS_REPO environment variable.
DEFAULT_REPO = os.environ.get("AUTOCLIP_MODELS_REPO", "shubhdotai/autoclip")
DEFAULT_REVISION = "main"

MANIFEST = Path(__file__).resolve().parents[1] / "models/manifest.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_url(repo, revision, path):
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


def fetch(url, destination, expected_sha, label):
    """Download to a temporary file, verify, then move into place atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as out:
            temporary = Path(out.name)
            print(f"  downloading {label}")
            request = urllib.request.Request(url, headers={"User-Agent": "autoclip"})
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
        actual = digest(temporary)
        if actual != expected_sha:
            raise ValueError(
                f"{label}: checksum mismatch\n"
                f"  expected {expected_sha}\n  got      {actual}"
            )
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def install(asset, name, output, repo, revision, source):
    origin = asset.get("origin", {})
    done = skipped = 0
    for item in asset["files"]:
        relative = item["path"]
        target = output / relative

        if target.exists():
            if digest(target) != item["sha256"]:
                raise SystemExit(
                    f"{target}: checksum mismatch; move it aside before retrying"
                )
            skipped += 1
            continue

        if source == "upstream":
            url = origin.get(relative)
            if url is None:
                raise SystemExit(
                    f"{relative}: no upstream origin recorded; use --source hf"
                )
        else:
            url = hf_url(repo, revision, relative)

        try:
            fetch(url, target, item["sha256"], relative)
        except urllib.error.HTTPError as exc:
            hint = ""
            if exc.code in (401, 403, 404) and source == "hf":
                hint = (f"\n  Check that {repo!r} exists and is public, or pass "
                        f"--repo / set AUTOCLIP_MODELS_REPO. Before uploading, use --source upstream.")
            raise SystemExit(f"{relative}: HTTP {exc.code} from {url}{hint}") from exc
        except urllib.error.URLError as exc:
            raise SystemExit(f"{relative}: {exc.reason}") from exc
        done += 1

    total_mb = sum(item["size"] for item in asset["files"]) / 1e6
    state = "already present" if done == 0 else f"{done} file(s) fetched"
    print(f"OK {name:<18} {total_mb:7.2f} MB  {state}"
          + (f", {skipped} reused" if done and skipped else ""))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--include-yolo", action="store_true",
                   help="Add the YOLO face/person detector (137 MB, AGPL-3.0)")
    p.add_argument("--all", action="store_true", help="Fetch every asset")
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"Hugging Face model repo (default: {DEFAULT_REPO})")
    p.add_argument("--revision", default=DEFAULT_REVISION,
                   help="Branch, tag or commit to pull from")
    p.add_argument("--source", choices=["hf", "upstream"], default="hf",
                   help="Where to fetch from (default: the Hugging Face mirror)")
    p.add_argument("--list", action="store_true",
                   help="Show the available assets and exit")
    p.add_argument("--output", type=Path, default=Path("models"))
    args = p.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    assets = manifest["assets"]

    if args.list:
        for name, asset in assets.items():
            size = sum(item["size"] for item in asset["files"]) / 1e6
            print(f"{name:<18} {size:8.2f} MB  {asset['license']:<10} "
                  f"{asset['description']}")
        return

    wanted = {"asd"}
    if args.all or args.include_yolo:
        wanted.add("yolo")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, asset in assets.items():
        if asset["group"] not in wanted:
            continue
        install(asset, name, args.output, args.repo, args.revision, args.source)

    if "yolo" in wanted:
        print("\nThe YOLO detector is AGPL-3.0 and Ultralytics carries its own "
              "terms; both differ from AutoClip's MIT license.")


if __name__ == "__main__":
    main()
