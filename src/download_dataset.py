import logging
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from config import load_config


CONFIG = load_config()
DATASET_CONFIG = CONFIG.get("download_dataset", {})
DATASET_NAME = DATASET_CONFIG["dataset_name"]
TEST_CSV_NAME = DATASET_CONFIG["test_csv_name"]
TRAIN_CSV_NAME = DATASET_CONFIG["train_csv_name"]

logger = logging.getLogger(__name__)


def download_dataset(dataset: str, dest: Path) -> None:
    api = KaggleApi()

    try:
        api.authenticate()
    except Exception as e:
        raise RuntimeError("Kaggle API credentials not found") from e

    api.dataset_download_files(dataset, path=dest, unzip=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    data_path = project_root / "data" / "raw"

    if (
        data_path.exists()
        and (data_path / TEST_CSV_NAME).exists()
        and (data_path / TRAIN_CSV_NAME).exists()
    ):
        logger.info("Dataset already exists in %s", data_path)

    else:
        DATA_DIR = Path(data_path)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        download_dataset(DATASET_NAME, DATA_DIR)
