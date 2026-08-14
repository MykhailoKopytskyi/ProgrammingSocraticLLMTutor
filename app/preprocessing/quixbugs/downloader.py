from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from app.common.config import QUIX_BUGS_URL


class QuixBugsDownloader:
    DATASET_URL = QUIX_BUGS_URL
    ARCHIVE_PATH = Path("data/raw/quixbugs.zip")
    OUTPUT_DIRECTORY = Path("data/raw/quixbugs")

    _EXPECTED_CASES = 40

    def ensure_available(self) -> Path:
        if self._dataset_is_ready():
            print(f"QuixBugs is already available at {self.OUTPUT_DIRECTORY}")
            return self.OUTPUT_DIRECTORY

        self.ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)

        if self.OUTPUT_DIRECTORY.exists():
            shutil.rmtree(self.OUTPUT_DIRECTORY)

        self.OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

        self._download()
        self._extract()
        self.ARCHIVE_PATH.unlink(missing_ok=True)

        if not self._dataset_is_ready():
            raise RuntimeError("Downloaded QuixBugs archive has an unexpected layout")

        print(f"QuixBugs is ready at {self.OUTPUT_DIRECTORY}")
        return self.OUTPUT_DIRECTORY

    def _download(self) -> None:
        print("Downloading QuixBugs...")
        urlretrieve(
            self.DATASET_URL,
            str(self.ARCHIVE_PATH),
        )

    def _extract(self) -> None:
        print("Extracting QuixBugs...")
        with ZipFile(self.ARCHIVE_PATH, "r") as archive:
            archive.extractall(self.OUTPUT_DIRECTORY)

    def _dataset_is_ready(self) -> bool:
        root = self._find_dataset_root()
        if root is None:
            return False

        test_directory = root / "python_testcases"
        test_paths = sorted(test_directory.glob("test_*.py"))

        if len(test_paths) != self._EXPECTED_CASES:
            return False

        for test_path in test_paths:
            name = test_path.stem.removeprefix("test_")

            buggy_path = root / "python_programs" / f"{name}.py"
            correct_path = root / "correct_python_programs" / f"{name}.py"

            if not buggy_path.is_file() or not correct_path.is_file():
                return False

        return True

    def _find_dataset_root(self) -> Path | None:
        if not self.OUTPUT_DIRECTORY.is_dir():
            return None

        if (self.OUTPUT_DIRECTORY / "python_programs").is_dir():
            return self.OUTPUT_DIRECTORY

        matches = []

        for path in self.OUTPUT_DIRECTORY.rglob("python_programs"):
            candidate = path.parent

            if (candidate / "correct_python_programs").is_dir() and (
                candidate / "python_testcases"
            ).is_dir():
                matches.append(candidate)

        if len(matches) != 1:
            return None

        return matches[0]
