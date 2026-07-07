from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
RESOURCES_PATH = APP_ROOT.parent / "resources"
MIGRATIONS_PATH = RESOURCES_PATH / "migrations"
