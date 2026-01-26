from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import os
from dotenv import load_dotenv
import kaggle


def download_dataset(dataset: str, dest: Path) -> None:
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=dest, unzip=True)


if __name__ == "__main__":
    load_dotenv()
    KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    data_path = project_root / "data" / "raw" / "train.csv"
    
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        DATA_DIR = Path("data_path")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DATASET_NAME = "parisrohan/credit-score-classification"
        
        download_dataset(DATASET_NAME, DATA_DIR)
    else:
        print(f"Dataset already exists in {data_path}")