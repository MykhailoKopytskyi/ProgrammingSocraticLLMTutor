from __future__ import annotations

import re
from pathlib import Path

from ..common.models import BugAnnotation
from .base import DatasetLoader, RawBenchmarkCase


class MultiDebugLoader(DatasetLoader):
    """
    Loads TreeInstruct MULTI_DEBUG and normalizes it into RawBenchmarkCase.
    MULTI_DEBUG-specific parsing is completely encapsulated here.
    """

    _VARIANTS = (
        ("1bug-MULTI_BUG", 1),
        ("2bug-MULTI-BUG", 2),
        ("3bug-MULTI-BUG", 3),
    )

    def __init__(
        self,
        root: str | Path,
    ):
        self._root = self._resolve_dataset_root(Path(root))
        self._reference_directory = self._root / "2bug-MULTI-BUG" / "data_txts"

        if not self._reference_directory.is_dir():
            raise FileNotFoundError(
                "Could not find MULTI_DEBUG data_txts directory: "
                f"{self._reference_directory}"
            )

    def load(
        self,
        *,
        limit: int | None = None,
    ) -> list[RawBenchmarkCase]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")

        cases: list[RawBenchmarkCase] = []
        for directory_name, bug_count in self._VARIANTS:
            directory = self._root / directory_name

            if not directory.is_dir():
                raise FileNotFoundError(f"Missing MULTI_DEBUG directory: {directory}")

            for raw_path in sorted(directory.glob("*.py")):
                cases.append(
                    self._load_case(
                        raw_path=raw_path,
                        bug_count=bug_count,
                    )
                )
                if limit is not None and len(cases) >= limit:
                    return cases

        if not cases:
            raise ValueError("No MULTI_DEBUG cases were found")

        return cases

    def _load_case(
        self,
        *,
        raw_path: Path,
        bug_count: int,
    ) -> RawBenchmarkCase:
        if bug_count == 1:
            return self._load_one_bug_case(raw_path)

        return self._load_processed_case(
            raw_path=raw_path,
            bug_count=bug_count,
        )

    def _load_one_bug_case(
        self,
        raw_path: Path,
    ) -> RawBenchmarkCase:
        text = raw_path.read_text(encoding="utf-8")
        problem = self._clean_problem(
            self._extract_tag(
                text=text,
                tag="problem",
                source=raw_path,
            )
        )

        fixes = self._extract_tag(
            text=text,
            tag="bug_fixes",
            source=raw_path,
        )

        descriptions = self._extract_tag(
            text=text,
            tag="bug_desc",
            source=raw_path,
            required=False,
        )

        bugs = self._build_bugs(
            fixes=fixes,
            descriptions=descriptions,
            expected_count=1,
            source=raw_path,
        )

        reference_path = self._reference_directory / f"{raw_path.name}.txt"
        reference_text = reference_path.read_text(encoding="utf-8")
        correct_code = self._strip_line_numbers(
            self._extract_txt_field(
                text=reference_text,
                field="correct_code",
                source=reference_path,
            )
        )

        return RawBenchmarkCase(
            case_id=(f"multi_debug_1bug__{raw_path.stem}"),
            problem_statement=problem,
            buggy_code=self._extract_raw_code(
                text=text,
                source=raw_path,
            ),
            correct_code=correct_code,
            bugs=bugs,
            source=(f"MULTI_DEBUG:1bug/{raw_path.name}"),
        )

    def _load_processed_case(
        self,
        *,
        raw_path: Path,
        bug_count: int,
    ) -> RawBenchmarkCase:
        processed_path = raw_path.parent / "data_txts" / f"{raw_path.name}.txt"
        text = processed_path.read_text(encoding="utf-8")
        problem = self._extract_txt_field(
            text=text,
            field="problem",
            source=processed_path,
        )

        fixes = self._extract_txt_field(
            text=text,
            field="bug_fixes",
            source=processed_path,
        )

        descriptions = self._extract_txt_field(
            text=text,
            field="bug_desc",
            source=processed_path,
            required=False,
        )

        bugs = self._build_bugs(
            fixes=fixes,
            descriptions=descriptions,
            expected_count=bug_count,
            source=processed_path,
        )

        buggy_code = self._strip_line_numbers(
            self._extract_txt_field(
                text=text,
                field="buggy_code",
                source=processed_path,
            )
        )

        correct_code = self._strip_line_numbers(
            self._extract_txt_field(
                text=text,
                field="correct_code",
                source=processed_path,
            )
        )

        return RawBenchmarkCase(
            case_id=(f"multi_debug_{bug_count}bug__{raw_path.stem}"),
            problem_statement=problem,
            buggy_code=buggy_code,
            correct_code=correct_code,
            bugs=bugs,
            source=(f"MULTI_DEBUG:{bug_count}bug/{raw_path.name}"),
        )

    def _build_bugs(
        self,
        *,
        fixes: str,
        descriptions: str,
        expected_count: int,
        source: Path,
    ) -> tuple[BugAnnotation, ...]:
        fix_lines = self._non_empty_lines(fixes)
        if len(fix_lines) != expected_count:
            raise ValueError(
                f"{source}: expected {expected_count} bug fixes, found {len(fix_lines)}"
            )

        description_lines = self._non_empty_lines(descriptions)

        # MULTI_DEBUG bug_desc is inconsistent. If it cannot be aligned safely, use then corresponding authoritative fix as description.
        if len(description_lines) != expected_count:
            description_lines = fix_lines
        return tuple(
            BugAnnotation(
                bug_id=f"bug_{index}",
                description=description,
                fix=fix,
            )
            for index, (
                description,
                fix,
            ) in enumerate(
                zip(
                    description_lines,
                    fix_lines,
                    strict=True,
                ),
                start=1,
            )
        )

    @staticmethod
    def _resolve_dataset_root(
        root: Path,
    ) -> Path:
        if (root / "1bug-MULTI_BUG").is_dir():
            return root

        matches = list(root.rglob("1bug-MULTI_BUG"))

        if len(matches) != 1:
            raise ValueError(
                f"Could not uniquely locate TreeInstruct Dataset under {root}"
            )

        return matches[0].parent

    @staticmethod
    def _extract_tag(
        *,
        text: str,
        tag: str,
        source: Path,
        required: bool = True,
    ) -> str:
        match = re.search(
            rf"<{re.escape(tag)}>"
            rf"\s*(.*?)\s*"
            rf"</{re.escape(tag)}>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if match is None:
            if required:
                raise ValueError(f"{source}: missing <{tag}>")

            return ""

        value = match.group(1).strip()
        if required and not value:
            raise ValueError(f"{source}: <{tag}> is empty")

        return value

    @staticmethod
    def _extract_txt_field(
        *,
        text: str,
        field: str,
        source: Path,
        required: bool = True,
    ) -> str:
        match = re.search(
            rf"{re.escape(field)}:"
            rf"\s*---\s*"
            rf"{re.escape(field)}:"
            rf"\s*(.*?)\n---",
            text,
            flags=re.DOTALL,
        )

        if match is None:
            if required:
                raise ValueError(f"{source}: missing {field}")

            return ""

        value = match.group(1).strip()

        if required and not value:
            raise ValueError(f"{source}: {field} is empty")

        return value

    @staticmethod
    def _extract_raw_code(
        *,
        text: str,
        source: Path,
    ) -> str:
        marker = "class Solution"

        position = text.find(marker)

        if position == -1:
            raise ValueError(f"{source}: could not find 'class Solution'")

        return text[position:].strip() + "\n"

    @staticmethod
    def _strip_line_numbers(
        code: str,
    ) -> str:
        cleaned_lines: list[str] = []

        for line in code.splitlines():
            match = re.match(
                r"^\s*\d+\.\s?(.*)$",
                line,
            )

            cleaned_lines.append(match.group(1) if match else line)

        return "\n".join(cleaned_lines).strip() + "\n"

    @staticmethod
    def _non_empty_lines(
        value: str,
    ) -> list[str]:
        return [line.strip() for line in value.splitlines() if line.strip()]

    @staticmethod
    def _clean_problem(
        problem: str,
    ) -> str:
        lines = problem.splitlines()

        if lines and lines[0].startswith("Problem Link:"):
            lines = lines[1:]

        return "\n".join(lines).strip()
