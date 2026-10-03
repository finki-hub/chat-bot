from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager

import anyio


@asynccontextmanager
async def closing_stream[T](body: AsyncIterable[T]) -> AsyncIterator[AsyncIterator[T]]:
    """Close the consumed iterator in its task, preserving any primary exception."""
    iterator = aiter(body)
    failed = False
    try:
        yield iterator
    except BaseException:
        failed = True
        raise
    finally:
        close = getattr(iterator, "aclose", None)
        if callable(close):
            try:
                with anyio.CancelScope(shield=True):
                    await close()
            except BaseException:
                if not failed:
                    raise
