from asyncio import Queue, QueueEmpty, Semaphore
from typing import Any

from io_chains._internal.sentinel import EndOfStream


class Buffer:
    """
    FIFO buffer whose capacity bounds *data* items only.

    Control signals (EndOfStream) are enqueued without waiting for capacity,
    so end-of-stream can always be delivered — even into a full buffer whose
    consumer has died. Without this, a cancelled link's EOS could block
    forever on a bounded downstream queue.

    maxsize <= 0 means unbounded (matching asyncio.Queue).
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: Queue = Queue()
        self._capacity: Semaphore | None = Semaphore(maxsize) if maxsize > 0 else None

    async def put(self, item: Any) -> None:
        if isinstance(item, EndOfStream):
            self._queue.put_nowait(item)
            return
        if self._capacity is not None:
            await self._capacity.acquire()
        self._queue.put_nowait(item)

    async def get(self) -> Any:
        return self._release(await self._queue.get())

    def get_nowait(self) -> Any:
        return self._release(self._queue.get_nowait())

    def _release(self, item: Any) -> Any:
        if self._capacity is not None and not isinstance(item, EndOfStream):
            self._capacity.release()
        return item

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def drain(self) -> None:
        while True:
            try:
                self.get_nowait()
            except QueueEmpty:
                return
