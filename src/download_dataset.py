from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import os
from dotenv import load_dotenv
import kaggle


def download_dataset(dataset: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.iterdir()):
        print(f"Dataset already exists in {dest}")
        return

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset, path=dest, unzip=True)


if __name__ == "__main__":
    load_dotenv()
    KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")

    DATA_DIR = Path("data/raw")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_NAME = "parisrohan/credit-score-classification"

    download_dataset(DATASET_NAME, DATA_DIR)
