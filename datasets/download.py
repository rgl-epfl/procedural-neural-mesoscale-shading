"""Download the released captures.

Each material is one archive holding ``<material>/{dataset.json, *.exr}``.
Archives are kept in ``<root>/.cache/`` and extracted into ``<root>/<material>``, with
``root`` defaulting to ``$MESOSCALE_DATA`` or ``~/.cache/mesoscale``. ``Photos``
downloads a missing material automatically; from the command line::

    python -m datasets.download                 # all materials
    python -m datasets.download alu fabric_hex --root /path/to/data
"""

from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path

import tqdm

URLS = {
    "alu": "https://files.mitsuba-renderer.org/doering2026/mesoscale/dataset/alu.zip",
    "fabric_black": "https://files.mitsuba-renderer.org/doering2026/mesoscale/dataset/fabric_black.zip",
    "fabric_blue": "https://files.mitsuba-renderer.org/doering2026/mesoscale/dataset/fabric_blue.zip",
    "fabric_gold": "https://files.mitsuba-renderer.org/doering2026/mesoscale/dataset/fabric_gold.zip",
    "fabric_grey": "https://files.mitsuba-renderer.org/doering2026/mesoscale/dataset/fabric_grey.zip",
    "fabric_hex": "https://files.mitsuba-renderer.org/doering2026/mesoscale/dataset/fabric_hex.zip",
    "fancy_fabric": "https://files.mitsuba-renderer.org/doering2026/mesoscale/dataset/fancy_fabric.zip",
    "fancy_paint": "https://files.mitsuba-renderer.org/doering2026/mesoscale/dataset/fancy_paint.zip",
    "glitter": "https://files.mitsuba-renderer.org/doering2026/mesoscale/dataset/glitter.zip",
}


def default_root() -> Path:
    """$MESOSCALE_DATA, or mesoscale/ in the user's cache folder."""
    if os.environ.get("MESOSCALE_DATA"):
        return Path(os.environ["MESOSCALE_DATA"]).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser() / "mesoscale"


def _fetch(url: str, dst: Path) -> None:
    """Download url to dst, resuming a partial download in dst.part."""
    part = dst.with_name(dst.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(request) as response:
        if done and response.status != 206:  # the server ignored the range
            done = 0
        length = response.headers.get("Content-Length")
        total = done + int(length) if length else None
        with open(part, "ab" if done else "wb") as f, tqdm.tqdm(
            total=total, initial=done, unit="B", unit_scale=True, desc=dst.name
        ) as bar:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    if total and part.stat().st_size != total:
        raise OSError(f"incomplete download of {url}; run again to resume")
    part.replace(dst)


def download(material: str, root: str | os.PathLike | None = None) -> Path:
    """Download and extract one material into root (default: default_root()); return its folder.

    Does nothing if root/<material>/dataset.json already exists, and reuses the
    archive in root/.cache/ if it was downloaded before."""
    root = Path(root).expanduser() if root else default_root()
    folder = root / material
    if (folder / "dataset.json").exists():
        return folder
    if material not in URLS:
        raise ValueError(f"unknown material {material!r}; expected one of {', '.join(URLS)}")
    archive = root / ".cache" / f"{material}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        _fetch(URLS[material], archive)
    with zipfile.ZipFile(archive) as z:
        for member in tqdm.tqdm(z.infolist(), desc=f"Extracting {material}", unit="file"):
            z.extract(member, root)
    if not (folder / "dataset.json").exists():
        raise OSError(f"{archive} does not contain {material}/dataset.json")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("materials", nargs="*", default=list(URLS), help="materials to download (default: all)")
    parser.add_argument("--root", help="destination folder (default: $MESOSCALE_DATA or ~/.cache/mesoscale)")
    args = parser.parse_args()
    for material in args.materials:
        print(download(material, args.root))


if __name__ == "__main__":
    main()
