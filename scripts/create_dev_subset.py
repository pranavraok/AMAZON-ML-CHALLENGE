"""Run development-subset creation without requiring an editable install."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from business_entity_resolution.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["create-dev-subset", *sys.argv[1:]]))
