from __future__ import annotations

import ast
import json
from pathlib import Path
from pprint import pformat

from ..raw_benchmark_case import RawBenchmarkCase


class QuixBugsParser:
    def parse_program(
        self,
        *,
        name: str,
        buggy_text: str,
        correct_text: str,
        tests: str,
        buggy_path: Path,
        correct_path: Path,
    ) -> RawBenchmarkCase:
        problem = self._clean_problem(self._extract_problem(buggy_text, buggy_path))

        buggy_code = self._extract_raw_code(
            buggy_text,
            buggy_path,
        )
        correct_code = self._extract_raw_code(
            correct_text,
            correct_path,
        )

        if buggy_code == correct_code:
            raise ValueError(f"{buggy_path}: buggy and correct code are identical")

        return RawBenchmarkCase(
            case_id=f"quixbugs__{name}",
            problem_statement=problem,
            buggy_code=buggy_code,
            correct_code=correct_code,
            bugs=(),
            source=f"QUIXBUGS:{name}",
            tests=tests,
        )

    def parse_json_tests(
        self,
        *,
        name: str,
        text: str,
        json_text: str,
        source: Path,
    ) -> str:
        text = self._replace_solution_import(
            text=text,
            name=name,
            source=source,
        )

        loader_import = "from load_testdata import load_json_testcases\n"

        if loader_import not in text:
            raise ValueError(f"{source}: missing load_testdata import")

        text = text.replace(
            loader_import,
            "",
            1,
        )

        testdata = []
        for line in self._normalize_newlines(json_text).splitlines():
            if line.strip():
                testdata.append(json.loads(line))

        loader_call = f"testdata = load_json_testcases({name}.__name__)"
        if loader_call not in text:
            raise ValueError(f"{source}: missing testdata load")

        formatted_testdata = pformat(
            testdata,
            width=88,
            sort_dicts=False,
        )

        text = text.replace(
            loader_call,
            f"testdata = {formatted_testdata}",
            1,
        )

        text = text.replace(
            "if not pytest.run_slow and ",
            "if ",
        )

        return text.strip() + "\n"

    def parse_python_tests(
        self,
        *,
        name: str,
        text: str,
        node_text: str,
        source: Path,
        node_source: Path,
    ) -> str:
        text = self._replace_solution_import(
            text=text,
            name=name,
            source=source,
        )

        node_import = "from node import Node\n"

        if node_import in text:
            if not node_text.strip():
                raise ValueError(f"{source}: Node helper is required")

            text = text.replace(
                node_import,
                "",
                1,
            )

            node_code = self._extract_raw_code(
                node_text,
                node_source,
            )

            text = node_code.rstrip() + "\n\n" + text.lstrip()

        return text.strip() + "\n"

    def _replace_solution_import(
        self,
        *,
        text: str,
        name: str,
        source: Path,
    ) -> str:
        text = self._normalize_newlines(text)

        import_block = (
            f"if pytest.use_correct:\n"
            f"    from correct_python_programs.{name} import {name}\n"
            "else:\n"
            f"    from python_programs.{name} import {name}\n"
        )

        if import_block not in text:
            raise ValueError(f"{source}: could not find QuixBugs solution import block")

        return text.replace(
            import_block,
            f"from solution import {name}\n",
            1,
        )

    def _extract_problem(
        self,
        text: str,
        source: Path,
    ) -> str:
        _, problem = self._trailing_string(
            text=text,
            source=source,
            required=True,
        )
        return problem

    def _extract_raw_code(
        self,
        text: str,
        source: Path,
    ) -> str:
        text = self._normalize_newlines(text)

        node, _ = self._trailing_string(
            text=text,
            source=source,
            required=False,
        )

        if node is not None:
            lines = text.splitlines()
            text = "\n".join(lines[: node.lineno - 1])

        value = text.strip()
        if not value:
            raise ValueError(f"{source}: code is empty")
        return value + "\n"

    def _trailing_string(
        self,
        *,
        text: str,
        source: Path,
        required: bool,
    ) -> tuple[ast.Expr | None, str]:
        text = self._normalize_newlines(text)

        tree = ast.parse(
            text,
            filename=str(source),
        )

        if tree.body:
            node = tree.body[-1]

            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return node, node.value.value

        if required:
            raise ValueError(f"{source}: missing trailing problem description")
        return None, ""

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _clean_problem(problem: str) -> str:
        return problem.strip()
