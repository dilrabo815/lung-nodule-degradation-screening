"""Download the LUNA25 public annotation CSV, nodule-block archives and the
Kinetics I3D starting weights.

Sources (see proposal refs [4], [5], [9]):
  * annotations  Zenodo 10.5281/zenodo.14673658   (CC BY-NC 4.0)
  * nodule blocks Zenodo 10.5281/zenodo.14223624  (CC BY 4.0), 2 split parts, ~6.8 GB
  * I3D weights   DIAGNijmegen/luna25-baseline-public resources/model_rgb.pth (Apache 2.0)

The nodule-block archive is a 7-Zip *split* zip (.zip.001, .zip.002). The parts
are plain byte splits, so ``cat part1 part2 > all.zip`` followed by unzip also
works when 7z is missing.

Usage:  python -m lungdeg.data.download --root /content/luna25
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ANNOT_URL = ("https://zenodo.org/records/14673658/files/"
             "LUNA25_Public_Training_Development_Data.csv?download=1")
ANNOT_MD5 = "1349fcd9fdd120a990f43a2301f1609b"
BLOCK_URLS = [f"https://zenodo.org/records/14223624/files/luna25_nodule_blocks.zip.00{i}?download=1"
              for i in (1, 2)]
I3D_URL = ("https://media.githubusercontent.com/media/DIAGNijmegen/"
           "luna25-baseline-public/main/resources/model_rgb.pth")


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def fetch(url: str, dest: Path) -> Path:
    """Resumable download with curl (falls back to urllib)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not dest.with_suffix(dest.suffix + ".part").exists():
        print(f"[skip] {dest.name} already present ({dest.stat().st_size/1e9:.2f} GB)")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    if shutil.which("curl"):
        _run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(tmp), url])
    else:  # pragma: no cover
        import urllib.request
        urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_blocks(raw: Path, out: Path) -> Path:
    """Extract the split archive and return the folder holding image/ and metadata/."""
    if find_block_root(out) is not None:
        print("[skip] nodule blocks already extracted")
        return find_block_root(out)
    out.mkdir(parents=True, exist_ok=True)
    first = raw / "luna25_nodule_blocks.zip.001"
    if shutil.which("7z"):
        _run(["7z", "x", "-y", f"-o{out}", str(first)])
    else:
        joined = raw / "luna25_nodule_blocks_joined.zip"
        with open(joined, "wb") as w:
            for p in sorted(raw.glob("luna25_nodule_blocks.zip.0*")):
                with open(p, "rb") as r:
                    shutil.copyfileobj(r, w, 1 << 24)
        _run(["unzip", "-q", "-o", str(joined), "-d", str(out)])
        joined.unlink()
    root = find_block_root(out)
    if root is None:
        raise RuntimeError(f"Could not find image/ and metadata/ folders under {out}")
    return root


def find_block_root(base: Path) -> Path | None:
    if not base.exists():
        return None
    for cand in [base, *[p for p in base.rglob("*") if p.is_dir()]]:
        if (cand / "image").is_dir() and (cand / "metadata").is_dir():
            return cand
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/luna25")
    ap.add_argument("--skip-blocks", action="store_true", help="CSV + weights only")
    a = ap.parse_args(argv)
    root = Path(a.root)
    raw = root / "raw"
    csv = fetch(ANNOT_URL, raw / "LUNA25_Public_Training_Development_Data.csv")
    got = md5(csv)
    print("annotation md5", got, "OK" if got == ANNOT_MD5 else "!! differs from Zenodo record")
    fetch(I3D_URL, raw / "model_rgb.pth")
    if not a.skip_blocks:
        for i, u in enumerate(BLOCK_URLS, 1):
            fetch(u, raw / f"luna25_nodule_blocks.zip.00{i}")
        blk = extract_blocks(raw, root / "nodule_blocks")
        n = len(list((blk / "image").glob("*.npy")))
        print(f"nodule blocks at {blk}  ({n} image files)")


if __name__ == "__main__":
    sys.exit(main())
