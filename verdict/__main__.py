"""CLI entry: ``python -m verdict`` runs the offline worked example.

Setup
-----
From a clone of this repository::

    pip install -e .
    python -m verdict

No API keys or Qdrant are required. Expected routing:

* Step A (mechanical rename) → ``bounded`` (R ≈ 0.3125)
* Step B (architectural boundary) → ``escalate`` (R ≈ 0.7175)
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    """Execute ``examples/worked_example.py`` as ``__main__``."""
    example = Path(__file__).resolve().parents[1] / "examples" / "worked_example.py"
    runpy.run_path(str(example), run_name="__main__")


if __name__ == "__main__":
    main()
