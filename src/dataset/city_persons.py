import os

from cityscapesscripts.download import downloader

from base import Dataset


class CityPersons(Dataset):
    def __init__(self):
        super().__init__()
        self.datadir = "../storage/data/city_persons"
        self.download()
        self.labels = ["person"]

    def download(self):
        os.makedirs(self.datadir, exist_ok=True)
        if os.path.exists(f"{self.datadir}/leftImg8bit_trainvaltest.zip"):
            session = downloader.login()
            downloader.download_packages(
                session=session,
                package_names=[
                    "leftImg8bit_trainvaltest.zip",
                    "gtBbox_cityPersons_trainval.zip",
                ],
                destination_path=self.datadir,
            )
