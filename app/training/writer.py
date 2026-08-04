from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")
