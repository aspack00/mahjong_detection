from pathlib import Path


DATASET_VERSION: int = 18


def get_dataset_root_path() -> Path:
    """
    Get root path of the Mahjong detection dataset for training.

    The path naming should match the folder created by Roboflow download.
    """
    return Path(f"Mahjong_detect-{DATASET_VERSION}")

