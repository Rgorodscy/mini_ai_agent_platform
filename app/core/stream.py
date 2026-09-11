"""
Bridges a blocking generator to an HTTP response, and formats SSE frames.

Why a producer thread rather than just yielding from the request handler:
usage accounting lives in a ContextVar (see core/usage.py), and Starlette
iterates a synchronous generator by handing each `next()` to a worker
thread with a *copy* of the caller's context. A `ContextVar.set()` made
before one yield is therefore invisible after it, so `track_usage()` around
a generator body silently measures nothing.

Running the whole execution inside one thread gives it one context for its
entire life, which fixes that and also decouples the agent's progress from
how fast the client reads: the queue is bounded, so a slow reader applies
backpressure instead of letting events pile up without limit.
"""

import json
import queue
import threading
from typing import Any, Callable, Iterator

from app.logger import get_logger

logger = get_logger(__name__)

# Enough slack that a normal run never blocks, small enough that a client
# which stops reading cannot hold unbounded memory.
QUEUE_SIZE = 100

# How long a put may block on a full queue before the producer gives up on
# a client that has stopped reading.
PUT_TIMEOUT_SECONDS = 30

_DONE = object()


def format_sse(event: str, data: Any) -> str:
    """
    One Server-Sent Events frame.

    `data` is JSON on a single line: a raw newline inside the data field
    would be read as a field separator and truncate the event.
    """
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def run_in_thread(
    produce: Callable[[Callable[[Any], None]], None],
) -> Iterator[Any]:
    """
    Runs `produce` in a thread, yielding whatever it emits.

    `produce` is called with an `emit` callable. Exceptions are forwarded to
    the consumer rather than dying silently in the worker thread, so the
    caller can turn them into an error event.
    """
    events: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
    failure: list[BaseException] = []

    def emit(item: Any) -> None:
        try:
            events.put(item, timeout=PUT_TIMEOUT_SECONDS)
        except queue.Full:
            raise BrokenPipeError("Client stopped reading the stream.")

    def worker() -> None:
        try:
            produce(emit)
        except BaseException as e:  # noqa: BLE001 — forwarded below
            failure.append(e)
        finally:
            # Sentinel goes in unconditionally: without it a consumer
            # blocks on get() forever when the producer dies.
            events.put(_DONE)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while True:
            item = events.get()
            if item is _DONE:
                break
            yield item
    finally:
        thread.join(timeout=5)

    if failure:
        raise failure[0]
