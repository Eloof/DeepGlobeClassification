import logging

from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleDatasetDownloader:
    """
    A class for downloading datasets from Kaggle.
    """

    def __init__(self, dataset_name: str, destination_path: str = "./") -> None:
        """
        Initialize the KaggleDatasetDownloader.

        Args:
            dataset_name: The Kaggle dataset name in the format "username/dataset-name".
            destination_path: (optional): The destination directory for downloading the dataset. Defaults to "./".
        """
        self.dataset_name = dataset_name
        self.destination_path = destination_path
        self.api = KaggleApi()
        self.api.authenticate()

    def download_dataset(self):
        """

        Download and unzip the Kaggle dataset.

        """
        try:
            logging.info(f"Downloading dataset '{self.dataset_name}'...")
            self.api.dataset_download_files(
                self.dataset_name, path=self.destination_path, unzip=True
            )

            logging.info(
                f"Dataset '{self.dataset_name}' successfully downloaded and unzipped in: {self.destination_path}."
            )
        except Exception as e:
            logging.error(f"Error downloading dataset: {str(e)}")




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    downloader = KaggleDatasetDownloader(
        "balraj98/deepglobe-land-cover-classification-dataset", "./Data"
    )

    downloader.download_dataset()
