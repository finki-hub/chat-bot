import anyio

from app.data.connection import Database
from app.utils.settings import Settings


async def main() -> None:
    """
    Connects to the database and runs all schema migrations.
    """
    settings = Settings()
    db = Database(dsn=settings.DATABASE_URL)

    await db.init()
    try:
        await db.run_migrations()
    finally:
        await db.disconnect()


if __name__ == "__main__":
    anyio.run(main)
