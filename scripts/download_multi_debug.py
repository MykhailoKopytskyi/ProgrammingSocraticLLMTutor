from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import gdown
from app.datasets.multi_debug_loader import MultiDebugLoader


class MultiDebugDownloadApp:
    DATASET_URL = (
        "https://drive.google.com/file/d/"
        "1ZFc7XfPoOShD3KZef3Z1DVHJU1fj_1LN/"
        "view?usp=sharing"
    )

    ARCHIVE_PATH = Path("data/raw/tree_instruct_multi_debug_dataset.zip")

    OUTPUT_DIRECTORY = Path("data/raw/multi_debug")
    EXPECTED_CASE_COUNT = 150

    def run(self) -> None:
        if self._dataset_is_ready():
            print(f"MULTI_DEBUG is already available at {self.OUTPUT_DIRECTORY}")
            return

        self.ARCHIVE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.OUTPUT_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._download()
        self._extract()
        self._validate()

        self.ARCHIVE_PATH.unlink(missing_ok=True)

        print(f"MULTI_DEBUG is ready at {self.OUTPUT_DIRECTORY}")

    def _download(self) -> None:
        print("Downloading MULTI_DEBUG...")

        downloaded_path = gdown.download(
            url=self.DATASET_URL,
            output=str(self.ARCHIVE_PATH),
            quiet=False,
            fuzzy=True,
        )

        if downloaded_path is None:
            raise RuntimeError("MULTI_DEBUG download failed.")

    def _extract(self) -> None:
        print("Extracting MULTI_DEBUG...")

        with ZipFile(
            self.ARCHIVE_PATH,
            "r",
        ) as archive:
            archive.extractall(self.OUTPUT_DIRECTORY)

    def _validate(self) -> None:
        print("Validating MULTI_DEBUG...")

        cases = MultiDebugLoader(self.OUTPUT_DIRECTORY).load()

        if len(cases) != self.EXPECTED_CASE_COUNT:
            raise RuntimeError(
                "Unexpected MULTI_DEBUG case count: "
                f"expected {self.EXPECTED_CASE_COUNT}, found {len(cases)}."
            )

    def _dataset_is_ready(self) -> bool:
        if not self.OUTPUT_DIRECTORY.exists():
            return False

        try:
            cases = MultiDebugLoader(self.OUTPUT_DIRECTORY).load()

            return len(cases) == self.EXPECTED_CASE_COUNT

        except (
            FileNotFoundError,
            ValueError,
        ):
            return False


if __name__ == "__main__":
    MultiDebugDownloadApp().run()
