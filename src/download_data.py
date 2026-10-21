"""
Step 1 of the pipeline: download raw ATP match files.

Data source: Jeff Sackmann's tennis_atp repository
  https://github.com/JeffSackmann/tennis_atp
Licensed CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike).

Note: as of late 2026 the original repo returns 404, so the script falls back
to archival mirrors of the same files (byte-identical; verified by SHA-1 for
2015, 2020 and 2024 against an independent fork).

Usage:
    python src/download_data.py
"""

from pathlib import Path
import shutil
import ssl
import urllib.error
import urllib.request

# Some Python installs (notably python.org builds on macOS) ship without system
# CA certificates. Use certifi's bundle when available so HTTPS verification works.
try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

# --- Configuration -----------------------------------------------------------
YEARS = range(2015, 2025)  # 2015 through 2024 inclusive
# Tried in order; the first URL that responds wins.
URL_TEMPLATES = [
    # Original source
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
    # Archival mirror of Sackmann's datasets
    "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp/atp_matches_{year}.csv",
    # Direct fork of JeffSackmann/tennis_atp
    "https://raw.githubusercontent.com/vishaan2010-dotcom/tennis_atp/master/atp_matches_{year}.csv",
]

# Resolve paths relative to the project root so the script works from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_year(year: int, force: bool = False) -> Path:
    """Download one season's match file into data/raw/ (skips if already present)."""
    dest = RAW_DIR / f"atp_matches_{year}.csv"
    if dest.exists() and not force:
        print(f"  {year}: already downloaded ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    for template in URL_TEMPLATES:
        url = template.format(year=year)
        try:
            with urllib.request.urlopen(url, context=SSL_CONTEXT, timeout=60) as resp:
                # Write to a temp file first so a failed download never leaves a partial CSV.
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(resp, f)
            tmp.replace(dest)
            print(f"  {year}: saved from {url} ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except urllib.error.URLError as err:  # HTTPError is a subclass
            print(f"  {year}: {url} failed ({err}); trying next source")

    raise RuntimeError(f"Could not download {year} from any source")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ATP matches {YEARS.start}-{YEARS.stop - 1} to {RAW_DIR}")
    for year in YEARS:
        download_year(year)
    print("Done.")


if __name__ == "__main__":
    main()
