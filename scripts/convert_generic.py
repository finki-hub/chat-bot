import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path("./scripts").resolve()


def script_file(name: str, suffix: str) -> Path:
    """A file directly inside ./scripts; any other path is refused."""
    path = (SCRIPTS_DIR / f"{name}{suffix}").resolve()
    if path.parent != SCRIPTS_DIR:
        raise SystemExit(f"{name!r} must name a file in {SCRIPTS_DIR}")
    return path


def convert_data(table: str, data: list) -> str:
    queries = []

    for item in data:
        # dynamically infer all keys from item and add them to query
        keys = ", ".join(item.keys())
        values = ", ".join([f"'{value}'" for value in item.values()])
        query = f"""INSERT INTO {table} ({keys})
        VALUES ({values});
        """

        queries.append(query)

    return "\n".join(queries)


if __name__ == "__main__":
    table_name = sys.argv[1]
    file = sys.argv[2]

    with script_file(file, ".json").open(encoding="utf-8") as f:
        data = json.load(f)

    with script_file(file, ".sql").open("w", encoding="utf-8") as f:
        f.write(convert_data(table_name, data))
