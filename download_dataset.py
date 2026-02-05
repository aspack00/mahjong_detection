from utils import download_data
from config import DATASET_VERSION


def main() -> None:
    """
    Download Mahjong detection dataset from Roboflow.

    The version number should match the dataset version configured
    in the Roboflow project and training notebook.
    """
    download_data(version=DATASET_VERSION)


if __name__ == "__main__":
    main()

