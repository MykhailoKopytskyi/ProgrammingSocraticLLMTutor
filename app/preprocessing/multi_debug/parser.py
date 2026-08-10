from __future__ import annotations

import re
from pathlib import Path

from ...common.models import BugAnnotation
from ..raw_benchmark_case import RawBenchmarkCase


class MultiDebugParser:
    def parse_one_bug(
        self,
        *,
        raw_text: str,
        reference_text: str,
        raw_path: Path,
        reference_path: Path,
    ) -> RawBenchmarkCase:
        problem = self._clean_problem(
            self._extract_tag(text=raw_text, tag="problem", source=raw_path)
        )
        fixes = self._extract_tag(text=raw_text, tag="bug_fixes", source=raw_path)
        descriptions = self._extract_tag(
            text=raw_text,
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
        correct_code = self._strip_line_numbers(
            self._extract_txt_field(
                text=reference_text,
                field="correct_code",
                source=reference_path,
            )
        )

        return RawBenchmarkCase(
            case_id=f"multi_debug_1bug__{raw_path.stem}",
            problem_statement=problem,
            buggy_code=self._extract_raw_code(raw_text, raw_path),
            correct_code=correct_code,
            bugs=bugs,
            source=f"MULTI_DEBUG:1bug/{raw_path.name}",
        )

    def parse_processed(
        self,
        *,
        text: str,
        raw_path: Path,
        processed_path: Path,
        bug_count: int,
    ) -> RawBenchmarkCase:
        problem = self._extract_txt_field(text=text, field="problem", source=processed_path)
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
            self._extract_txt_field(text=text, field="buggy_code", source=processed_path)
        )
        correct_code = self._strip_line_numbers(
            self._extract_txt_field(text=text, field="correct_code", source=processed_path)
        )

        return RawBenchmarkCase(
            case_id=f"multi_debug_{bug_count}bug__{raw_path.stem}",
            problem_statement=problem,
            buggy_code=buggy_code,
            correct_code=correct_code,
            bugs=bugs,
            source=f"MULTI_DEBUG:{bug_count}bug/{raw_path.name}",
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
        # bug_desc is inconsistent in MULTI_DEBUG. The fix text is authoritative.
        if len(description_lines) != expected_count:
            description_lines = fix_lines

        bugs: list[BugAnnotation] = []
        pairs = zip(description_lines, fix_lines, strict=True)
        for index, (description, fix) in enumerate(pairs, start=1):
            bugs.append(
                BugAnnotation(
                    bug_id=f"bug_{index}",
                    description=description,
                    fix=fix,
                )
            )
        return tuple(bugs)

    @staticmethod
    def _extract_tag(
        text: str,
        tag: str,
        source: Path,
        required: bool = True,
    ) -> str:
        match = re.search(
            rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
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
        text: str,
        field: str,
        source: Path,
        required: bool = True,
    ) -> str:
        match = re.search(
            rf"{re.escape(field)}:\s*---\s*{re.escape(field)}:\s*(.*?)\n---",
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
    def _extract_raw_code(text: str, source: Path) -> str:
        marker = "class Solution"
        position = text.find(marker)
        if position == -1:
            raise ValueError(f"{source}: could not find 'class Solution'")
        return text[position:].strip() + "\n"

    @staticmethod
    def _strip_line_numbers(code: str) -> str:
        cleaned_lines: list[str] = []
        for line in code.splitlines():
            match = re.match(r"^\s*\d+\.\s?(.*)$", line)
            cleaned_lines.append(match.group(1) if match else line)
        return "\n".join(cleaned_lines).strip() + "\n"

    @staticmethod
    def _non_empty_lines(value: str) -> list[str]:
        lines = []
        for line in value.splitlines():
            line = line.strip()
            if line:
                lines.append(line)
        return lines

    @staticmethod
    def _clean_problem(problem: str) -> str:
        lines = problem.splitlines()
        if lines and lines[0].startswith("Problem Link:"):
            lines = lines[1:]
        return "\n".join(lines).strip()
