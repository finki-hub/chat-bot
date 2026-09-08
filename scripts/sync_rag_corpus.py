"""Compatibility wrapper for the packaged ``app.sync_rag_corpus`` entrypoint."""

# ruff: file-ignore[INP001] - compatibility wrapper outside the app package

import sys
from pathlib import Path

if (Path.cwd() / "app").is_dir() and str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from app.sync_rag_corpus import main

if __name__ == "__main__":
    raise SystemExit(main())
