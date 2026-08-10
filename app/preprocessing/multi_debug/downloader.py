from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZipFile

import gdown


class MultiDebugDownloader:
    DATASET_URL = (
        "https://drive.google.com/file/d/"
        "1ZFc7XfPoOShD3KZef3Z1DVHJU1fj_1LN/view?usp=sharing"
    )
    ARCHIVE_PATH = Path("data/raw/tree_instruct_multi_debug_dataset.zip")
    OUTPUT_DIRECTORY = Path("data/raw/multi_debug")

    _EXPECTED_VARIANTS = (
        "1bug-MULTI_BUG",
        "2bug-MULTI-BUG",
        "3bug-MULTI-BUG",
    )
    _EXPECTED_CASES_PER_VARIANT = 50

    def ensure_available(self) -> Path:
        if self._dataset_is_ready():
            print(f"MULTI_DEBUG is already available at {self.OUTPUT_DIRECTORY}")
            return self.OUTPUT_DIRECTORY

        self.ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if self.OUTPUT_DIRECTORY.exists():
            shutil.rmtree(self.OUTPUT_DIRECTORY)
        self.OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

        self._download()
        self._extract()
        self.ARCHIVE_PATH.unlink(missing_ok=True)

        if not self._dataset_is_ready():
            raise RuntimeError("Downloaded MULTI_DEBUG archive has an unexpected layout")

        print(f"MULTI_DEBUG is ready at {self.OUTPUT_DIRECTORY}")
        return self.OUTPUT_DIRECTORY

    def _download(self) -> None:
        print("Downloading MULTI_DEBUG...")
        downloaded_path = gdown.download(
            url=self.DATASET_URL,
            output=str(self.ARCHIVE_PATH),
            quiet=False,
            fuzzy=True,
        )
        if downloaded_path is None:
            raise RuntimeError("MULTI_DEBUG download failed")

    def _extract(self) -> None:
        print("Extracting MULTI_DEBUG...")
        with ZipFile(self.ARCHIVE_PATH, "r") as archive:
            archive.extractall(self.OUTPUT_DIRECTORY)

    def _dataset_is_ready(self) -> bool:
        if not self.OUTPUT_DIRECTORY.is_dir():
            return False

        roots = list(self.OUTPUT_DIRECTORY.rglob("1bug-MULTI_BUG"))
        if len(roots) != 1:
            return False
        root = roots[0].parent

        for directory in self._EXPECTED_VARIANTS:
            variant_path = root / directory
            if not variant_path.is_dir():
                return False
            if len(list(variant_path.glob("*.py"))) != self._EXPECTED_CASES_PER_VARIANT:
                return False

        for directory in ("2bug-MULTI-BUG", "3bug-MULTI-BUG"):
            data_txts = root / directory / "data_txts"
            if not data_txts.is_dir():
                return False
            if len(list(data_txts.glob("*.txt"))) != self._EXPECTED_CASES_PER_VARIANT:
                return False

        return True
