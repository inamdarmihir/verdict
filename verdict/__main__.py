"""CLI entry: ``python -m verdict`` runs the offline worked example."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "worked_example.py"
    runpy.run_path(str(example), run_name="__main__")


if __name__ == "__main__":
    main()
