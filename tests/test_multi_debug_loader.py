from pathlib import Path

from app.datasets.base import RawBenchmarkCase
from app.datasets.multi_debug_loader import MultiDebugLoader


def test_multi_debug_loader_normalizes_all_variants(tmp_path: Path):
    root = tmp_path / "TreeInstruct Dataset"
    one_dir = root / "1bug-MULTI_BUG"
    two_dir = root / "2bug-MULTI-BUG"
    three_dir = root / "3bug-MULTI-BUG"

    for directory in (one_dir, two_dir, three_dir):
        directory.mkdir(parents=True)

    (two_dir / "data_txts").mkdir()
    (three_dir / "data_txts").mkdir()

    raw_one = '''"""
<problem>
Return the input value.
</problem>
<bug_fixes>
Replace `return value + 1` with `return value`.
</bug_fixes>
<bug_desc>
The function adds one unnecessarily.
</bug_desc>
"""
class Solution:
    def solve(self, value):
        return value + 1
'''
    (one_dir / "example.py").write_text(raw_one, encoding="utf-8")

    processed_two = """example.py
problem: ---
problem:
Return the input value.
---

bug_fixes: ---
bug_fixes:
Replace `return value + 1` with `return value`.
Replace `value = int(value)` with `value = value`.
---

bug_desc: ---
bug_desc:
The return value is incremented.
The input is changed unnecessarily.
---

buggy_code: ---
buggy_code:
1. class Solution:
2.     def solve(self, value):
3.         value = int(value)
4.         return value + 1
---

correct_code: ---
correct_code:
1. class Solution:
2.     def solve(self, value):
3.         return value
---
"""
    (two_dir / "data_txts" / "example.py.txt").write_text(
        processed_two,
        encoding="utf-8",
    )
    (two_dir / "example.py").write_text("# source marker\n", encoding="utf-8")

    processed_three = """example.py
problem: ---
problem:
Return the input value.
---

bug_fixes: ---
bug_fixes:
Fix one.
Fix two.
Fix three.
---

bug_desc: ---
bug_desc:
---

buggy_code: ---
buggy_code:
1. class Solution:
2.     def solve(self, value):
3.         return value + 3
---

correct_code: ---
correct_code:
1. class Solution:
2.     def solve(self, value):
3.         return value
---
"""
    (three_dir / "data_txts" / "example.py.txt").write_text(
        processed_three,
        encoding="utf-8",
    )
    (three_dir / "example.py").write_text("# source marker\n", encoding="utf-8")

    loader = MultiDebugLoader(root)
    cases = loader.load()

    assert len(cases) == 3
    assert all(isinstance(case, RawBenchmarkCase) for case in cases)

    one_case, two_case, three_case = cases

    assert one_case.case_id == "multi_debug_1bug__example"
    assert one_case.correct_code == (
        "class Solution:\n    def solve(self, value):\n        return value\n"
    )
    assert len(one_case.bugs) == 1

    assert two_case.case_id == "multi_debug_2bug__example"
    assert len(two_case.bugs) == 2
    assert "value = int(value)" in two_case.buggy_code

    assert three_case.case_id == "multi_debug_3bug__example"
    assert len(three_case.bugs) == 3
    assert three_case.bugs[0].description == "Fix one."
