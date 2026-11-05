"""Download NHTSA ODI flat files into data/raw/nhtsa/ and unzip them.

Usage:
    python scripts/download_nhtsa.py                      # recalls + investigations + 2015-2019 complaints
    python scripts/download_nhtsa.py --complaints 2015-2019 2020-2024
    python scripts/download_nhtsa.py --complaints all     # full complaints file (358 MB zipped)
    python scripts/download_nhtsa.py --tsbs               # also technical service bulletins (2020-2024)

Source and data dictionaries: https://www.nhtsa.gov/nhtsa-datasets-and-apis
"""
import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

BASE = "https://static.nhtsa.gov/odi/ffdd"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "nhtsa"

SLICES = ["1995-1999", "2000-2004", "2005-2009", "2010-2014", "2015-2019", "2020-2024", "2025-2026"]
FILES = {
    "recalls": [f"{BASE}/rcl/FLAT_RCL_POST_2010.zip"],
    "investigations": [f"{BASE}/inv/FLAT_INV.zip"],
}
DICTIONARIES = [f"{BASE}/cmpl/CMPL.txt", f"{BASE}/rcl/RCL.txt", f"{BASE}/inv/INV.txt"]


def fetch(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / url.rsplit("/", 1)[-1]
    if out.exists():
        print(f"skip (already have): {out.name}")
        return out
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = out.with_suffix(out.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(out)
    return out


def unzip(path: Path) -> None:
    with zipfile.ZipFile(path) as z:
        z.extractall(path.parent)
        print(f"unzipped {path.name}: {', '.join(z.namelist())}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--complaints", nargs="+", default=["2015-2019"],
                   help=f"'all' or any of {SLICES}")
    p.add_argument("--tsbs", action="store_true", help="also download TSBs received 2020-2024")
    args = p.parse_args()

    files = dict(FILES)
    if args.complaints == ["all"]:
        files["complaints"] = [f"{BASE}/cmpl/FLAT_CMPL.zip"]
    else:
        bad = [s for s in args.complaints if s not in SLICES]
        if bad:
            p.error(f"unknown slice(s) {bad}; choose from {SLICES} or 'all'")
        files["complaints"] = [f"{BASE}/cmpl/COMPLAINTS_RECEIVED_{s}.zip" for s in args.complaints]
    if args.tsbs:
        files["tsbs"] = [f"{BASE}/tsbs/TSBS_RECEIVED_2020-2024.zip"]

    for name, urls in files.items():
        for url in urls:
            unzip(fetch(url, RAW / name))
    for url in DICTIONARIES:
        fetch(url, RAW / "dictionaries")
    print(f"\nDone. Raw files are in {RAW}")


if __name__ == "__main__":
    main()
