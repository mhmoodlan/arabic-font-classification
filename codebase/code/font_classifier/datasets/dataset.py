"""Dataset class to be extended by dataset-specific classes."""
from pathlib import Path
import os

from font_classifier import util

class Dataset:
    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[3] / "data"
    
    def load_or_generate_data(self):
        pass


def _download_raw_dataset(metadata):
    if os.path.exists(metadata["filename"]):
        return
    print(f"Downloading raw dataset from {metadata['url']}...")
    util.download_url(metadata["url"], metadata["filename"])
    print("Computing SHA-256...")
    sha256 = util.compute_sha256(metadata["filename"])
    if sha256 != metadata["sha256"]:
        raise ValueError("Downloaded data file SHA-256 does not match that listed in metadata document.")
