"""
Data Download Script for Disaster Tweets Project
Pull the labeled data from the image source (ucbrise/kaggle-nlp-disasters) into the data/ folder. 
Only download train.csv, since test.csv and sample_submission.csv are disabled in the assignment.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Web path: https://github.com/ucbrise/kaggle-nlp-disasters/tree/master/data
# Original file path: https://raw.githubusercontent.com/ucbrise/kaggle-nlp-disasters/master/data/
BASE_RAW_URL = "https://raw.githubusercontent.com/ucbrise/kaggle-nlp-disasters/master/data"

REQUIRED_FILES = ["train.csv"]


# OPTIONAL_FILES = ["test.csv", "sample_submission.csv"]


def download_file(url: str, dest_path: Path) -> None:

    # If the file already exists, skip.
    if dest_path.exists():
        print(f"⏭️  {dest_path.name} already exists, skip the downloading process.")
        return

    print(f"⬇️  Downloading from {url} ……")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"✅ Successfully downloaded from {dest_path}")
    except urllib.error.HTTPError as e:
        print(f"❌ Download fail (HTTP {e.code}): {e.reason}")
        print(f"   Please check your network connection, or verify whether the file exists in the mirror source.")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"❌ Download fail (Web Error): {e.reason}")
        print("   Please check the web connection")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unknown error occurred: {e}")
        sys.exit(1)


def main():
    # 1. create data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"📁 Data storage directory: {data_dir.absolute()}")

    # 2. download train.csv
    for filename in REQUIRED_FILES:
        file_url = f"{BASE_RAW_URL}/{filename}"
        dest_file = data_dir / filename
        download_file(file_url, dest_file)

    print("\n🎯Note: Only downloaded train.csv, because the course rules say test.csv and sample_submission.csv aren't used.")
    print("  If you really need test.csv for external testing, you can uncomment OPTIONAL_FILES in the code yourself.")

    # 4. Check if the key files really exist
    required_path = data_dir / "train.csv"
    if required_path.exists():
        file_size = required_path.stat().st_size
        print(f"✅ Data is ready! Size of train.csv: {file_size // 1024} KB")
    else:
        print("⚠️ Warning: train.csv failed to download successfully, please check the error message.")


if __name__ == "__main__":
    main()