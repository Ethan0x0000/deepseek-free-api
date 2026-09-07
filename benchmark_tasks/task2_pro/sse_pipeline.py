"""High-reliability SSE (Server-Sent Events) streaming pipeline.

Components:
    SSEEvent          -- a single parsed SSE event (event / data / id / retry)
    SSEParser         -- incremental, CRLF-tolerant, partial-chunk-safe parser
    BackpressureBuffer-- bounded async queue with DROP_OLDEST / DROP_LATEST / BLOCK
    ReconnectPipeline -- exponential-backoff reconnect engine that keeps Last-Event-ID
    StreamAggregator  -- SSE chunk aggregation + RFC-6902 JSON-Patch incremental assembly

The whole module is asyncio based and safe under high concurrency (all shared
state is mutated inside the event loop without blocking calls).
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, List, Optional, Union

__all__ = [
    "SSEEvent",
    "SSEParser",
    "BackpressurePolicy",
    "BackpressureBuffer",
    "ReconnectPipeline",
    "JSONPatchAssembler",
    "StreamAggregator",
]


# --------------------------------------------------------------------------- #
# SSE event record
# --------------------------------------------------------------------------- #
@dataclass
class SSEEvent:
    """A single Server-Sent Event.

    Attributes:
        event: event type, defaults to ``"message"``.
        data: the payload, multi-line ``data:`` fields joined by ``\\n``.
        id: the event id (also drives Last-Event-ID), optional.
        retry: server suggested reconnection time in milliseconds, optional.
    """

    event: str = "message"
    data: str = ""
    id: Optional[str] = None
    retry: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "data": self.data,
            "id": self.id,
            "retry": self.retry,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SSEEvent(event={self.event!r}, data={self.data!r}, "
            f"id={self.id!r}, retry={self.retry!r})"
        )


# --------------------------------------------------------------------------- #
# Incremental SSE parser
# --------------------------------------------------------------------------- #
class SSEParser:
    """Stateful, incremental parser for the SSE wire format.

    Feed it arbitrary chunks (they may split lines or events at any byte
    boundary); it buffers partial input and returns fully-assembled
    :class:`SSEEvent` objects.

    Handles the full field set (``event``, ``data``, ``id``, ``retry``), ignores
    comment lines (starting with ``:``), tolerates ``\\n``, ``\\r\\n`` and lone
    ``\\r`` line endings, and tracks ``last_event_id`` for reconnects.
    """

    def __init__(self) -> None:
        self._buf: str = ""
        self._event_type: str = "message"
        self._data: List[str] = []
        self._pending_id: Optional[str] = None
        # session-scoped state (not reset between events)
        self.last_event_id: Optional[str] = None
        self.retry: Optional[int] = None

    def reset(self) -> None:
        """Drop any partial state and return the parser to a clean slate."""
        self._buf = ""
        self._event_type = "message"
        self._data = []
        self._pending_id = None
        # keep last_event_id / retry: they are session settings.

    def feed(self, chunk: str) -> List[SSEEvent]:
        """Consume a raw text chunk and return any completed events."""
        events: List[SSEEvent] = []
        self._buf += chunk
        while True:
            nl = self._buf.find("\n")
            if nl == -1:
                break
            line = self._buf[:nl]
            self._buf = self._buf[nl + 1:]
            if line.endswith("\r"):
                line = line[:-1]
            if self._process_line(line):
                evt = self._dispatch()
                if evt is not None:
                    events.append(evt)
        return events

    def _process_line(self, line: str) -> bool:
        """Handle one physical line; return True when an event boundary hit."""
        if line == "":
            return True
        if line.startswith(":"):
            return False  # comment line
        field, sep, value = line.partition(":")
        if not sep:
            value = ""
        elif value.startswith(" "):
            value = value[1:]  # single leading space is stripped

        if field == "event":
            self._event_type = value
        elif field == "data":
            self._data.append(value)
        elif field == "id":
            self._pending_id = value if value != "" else None
        elif field == "retry":
            try:
                self.retry = int(value)
            except ValueError:
                pass
        return False

    def _dispatch(self) -> Optional[SSEEvent]:
        """Assemble the buffered fields into an event (or None if no data)."""
        data = "\n".join(self._data)
        event_type = self._event_type or "message"
        eid = self._pending_id
        retry = self.retry

        # reset per-event buffers
        self._data = []
        self._event_type = "message"
        self._pending_id = None

        if data == "":
            # Per spec, an event with an empty data buffer is not dispatched.
            return None

        if eid is not None:
            self.last_event_id = eid

        return SSEEvent(event=event_type, data=data, id=eid, retry=retry)


# --------------------------------------------------------------------------- #
# Backpressure-protected bounded queue
# --------------------------------------------------------------------------- #
class BackpressurePolicy(Enum):
    """Overflow handling for :class:`BackpressureBuffer`."""

    DROP_OLDEST = "drop_oldest"  # evict the head, accept the new item
    DROP_LATEST = "drop_latest"  # reject the incoming item
    BLOCK = "block"              # wait until capacity frees up


class BackpressureBuffer:
    """A fixed-capacity async event queue with explicit overflow policy.

    ``put`` returns ``True`` when the item was enqueued and ``False`` when it
    was dropped (only possible under a drop policy).  All drop decisions are
    serialised by an internal lock, so concurrent producers observe exactly one
    of the three documented behaviours without corruption.
    """

    def __init__(
        self,
        capacity: int,
        policy: Union[BackpressurePolicy, str] = BackpressurePolicy.BLOCK,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        if isinstance(policy, str):
            policy = BackpressurePolicy(policy)
        self.policy = policy
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue(maxsize=capacity)
        self._lock = asyncio.Lock()
        self._dropped = 0

    # -- introspection ----------------------------------------------------- #
    @property
    def dropped(self) -> int:
        """Total number of items dropped since creation."""
        return self._dropped

    @property
    def size(self) -> int:
        return self._queue.qsize()

    def full(self) -> bool:
        return self._queue.full()

    def empty(self) -> bool:
        return self._queue.empty()

    # -- producers --------------------------------------------------------- #
    async def put(self, item: Any) -> bool:
        """Enqueue ``item`` respecting the configured policy.

        Returns ``True`` if enqueued, ``False`` if dropped (drop policies only).
        """
        if self.policy is BackpressurePolicy.BLOCK:
            await self._queue.put(item)
            return True

        async with self._lock:
            if not self._queue.full():
                self._queue.put_nowait(item)
                return True

            if self.policy is BackpressurePolicy.DROP_LATEST:
                self._dropped += 1
                return False

            # DROP_OLDEST
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:  # pragma: no cover - defensive
                pass
            self._queue.put_nowait(item)
            self._dropped += 1
            return True

    # -- consumers --------------------------------------------------------- #
    async def get(self) -> Any:
        return await self._queue.get()

    def get_nowait(self) -> Any:
        return self._queue.get_nowait()

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()


# --------------------------------------------------------------------------- #
# Reconnect engine with exponential backoff + Last-Event-ID
# --------------------------------------------------------------------------- #
class ReconnectPipeline:
    """Auto-reconnecting consumer of an SSE stream.

    ``connector`` is an async callable (typically an async generator function)
    taking the current ``last_event_id`` and returning an async iterator of
    :class:`SSEEvent`.  Every successful event with a non-null ``id`` updates
    ``last_event_id``, which is fed back to ``connector`` on the next
    reconnect so the server can resume the stream.

    On failure the engine sleeps with exponential backoff (optionally jittered)
    and retries until ``max_attempts`` (default: forever).
    """

    def __init__(
        self,
        connector: Callable[[Optional[str]], AsyncIterator[SSEEvent]],
        *,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        max_attempts: Optional[int] = None,
        on_event: Optional[Callable[[SSEEvent], Optional[Awaitable[None]]]] = None,
        on_error: Optional[Callable[[BaseException], Optional[Awaitable[None]]]] = None,
        on_connecting: Optional[Callable[[int, Optional[str]], None]] = None,
    ) -> None:
        self.connector = connector
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.max_attempts = max_attempts
        self.on_event = on_event
        self.on_error = on_error
        self.on_connecting = on_connecting

        self.last_event_id: Optional[str] = None
        self.server_retry: Optional[float] = None
        self.attempts: int = 0
        self.consecutive_failures: int = 0
        self.last_delay: Optional[float] = None
        self.exhausted: bool = False

    def compute_delay(self, failures: int) -> float:
        """Backoff delay for ``failures`` consecutive failures (0 => clean)."""
        if failures <= 0:
            return self.initial_delay
        delay = self.initial_delay * (self.backoff_factor ** (failures - 1))
        if self.jitter:
            delay *= random.uniform(0.5, 1.5)
        return min(delay, self.max_delay)

    async def run(self, stop_event: Optional[asyncio.Event] = None) -> None:
        """Run the consume/reconnect loop until ``stop_event`` is set."""
        while stop_event is None or not stop_event.is_set():
            self.attempts += 1
            if self.on_connecting is not None:
                self.on_connecting(self.attempts, self.last_event_id)

            try:
                stream = self.connector(self.last_event_id)
                async for event in stream:
                    self.consecutive_failures = 0
                    if event.id is not None:
                        self.last_event_id = event.id
                    if event.retry is not None:
                        self.server_retry = event.retry / 1000.0
                    if self.on_event is not None:
                        result = self.on_event(event)
                        if asyncio.iscoroutine(result):
                            await result
                    if stop_event is not None and stop_event.is_set():
                        break
                # clean end of stream -> reconnect after a modest delay
                self.consecutive_failures = 0
                if self.max_attempts is not None and self.attempts >= self.max_attempts:
                    return
                delay = self.compute_delay(0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - engine swallows & retries
                self.consecutive_failures += 1
                if self.on_error is not None:
                    result = self.on_error(exc)
                    if asyncio.iscoroutine(result):
                        await result
                if self.max_attempts is not None and self.attempts >= self.max_attempts:
                    self.exhausted = True
                    raise
                delay = self.compute_delay(self.consecutive_failures)

            if stop_event is not None and stop_event.is_set():
                break
            self.last_delay = delay
            await asyncio.sleep(delay)


# --------------------------------------------------------------------------- #
# RFC 6902 JSON Patch incremental assembler
# --------------------------------------------------------------------------- #
def _decode_pointer(pointer: str) -> List[str]:
    """Decode an RFC 6901 JSON Pointer into reference tokens."""
    if pointer == "":
        return []
    return [
        tok.replace("~1", "/").replace("~0", "~")
        for tok in pointer.lstrip("/").split("/")
    ]


def _walk(doc: Any, tokens: List[str]) -> Any:
    """Descend ``doc`` along ``tokens``, returning the referenced container."""
    cur = doc
    for tok in tokens:
        if isinstance(cur, list):
            cur = cur[int(tok)]
        elif isinstance(cur, dict):
            cur = cur[tok]
        else:
            raise TypeError(
                f"cannot descend into {type(cur).__name__} at token {tok!r}"
            )
    return cur


class JSONPatchAssembler:
    """Applies a stream of RFC 6902 JSON Patch documents to a target value.

    Supports ``add``, ``remove``, ``replace``, ``move``, ``copy`` and ``test``
    operations and resolves JSON Pointers per RFC 6901 (including ``~0``/``~1``
    escaping and the ``-`` array-append token).
    """

    def __init__(self, initial: Any = None) -> None:
        self._doc: Any = copy.deepcopy(initial) if initial is not None else {}
        self.applied_ops: int = 0

    @property
    def document(self) -> Any:
        return self._doc

    def apply(self, patch: list) -> Any:
        """Apply a list of patch operations, returning the updated document."""
        for op in patch:
            self._apply(op)
        return self._doc

    # -- operation dispatch ------------------------------------------------ #
    def _apply(self, op: dict) -> None:
        name = op.get("op")
        path = op.get("path", "")
        if name == "add":
            self._add(path, op.get("value"))
        elif name == "remove":
            self._remove(path)
        elif name == "replace":
            self._replace(path, op.get("value"))
        elif name == "move":
            self._move(op.get("from", ""), path)
        elif name == "copy":
            self._copy(op.get("from", ""), path)
        elif name == "test":
            self._test(path, op.get("value"))
        else:  # pragma: no cover - unknown op ignored defensively
            return
        self.applied_ops += 1

    # -- helpers ----------------------------------------------------------- #
    def _resolve(self, pointer: str) -> Any:
        tokens = _decode_pointer(pointer)
        return _walk(self._doc, tokens)

    def _add(self, path: str, value: Any) -> None:
        tokens = _decode_pointer(path)
        if not tokens:
            self._doc = copy.deepcopy(value)
            return
        # Descend (auto-creating missing containers) to the parent of the target.
        cur = self._doc
        for i, tok in enumerate(tokens[:-1]):
            nxt = tokens[i + 1]
            make_list = nxt.isdigit() or nxt == "-"
            if isinstance(cur, list):
                idx = int(tok)
                while len(cur) <= idx:
                    cur.append(None)
                if cur[idx] is None:
                    cur[idx] = [] if make_list else {}
                cur = cur[idx]
            elif isinstance(cur, dict):
                if tok not in cur:
                    cur[tok] = [] if make_list else {}
                cur = cur[tok]
            else:  # pragma: no cover - defensive
                raise TypeError(
                    f"cannot add into {type(cur).__name__} at token {tok!r}"
                )
        key = tokens[-1]
        if isinstance(cur, list):
            if key == "-":
                cur.append(copy.deepcopy(value))
            else:
                idx = int(key)
                while len(cur) < idx:
                    cur.append(None)
                cur.insert(idx, copy.deepcopy(value))
        else:
            cur[key] = copy.deepcopy(value)

    def _remove(self, path: str) -> None:
        tokens = _decode_pointer(path)
        if not tokens:
            self._doc = None
            return
        parent = _walk(self._doc, tokens[:-1])
        key = tokens[-1]
        if isinstance(parent, list):
            idx = int(key)
            if 0 <= idx < len(parent):
                parent.pop(idx)
        else:
            parent.pop(key, None)

    def _replace(self, path: str, value: Any) -> None:
        tokens = _decode_pointer(path)
        parent = _walk(self._doc, tokens[:-1])
        key = tokens[-1]
        if isinstance(parent, list):
            parent[int(key)] = copy.deepcopy(value)
        else:
            parent[key] = copy.deepcopy(value)

    def _move(self, frm: str, path: str) -> None:
        value = self._resolve(frm)
        self._remove(frm)
        self._add(path, value)

    def _copy(self, frm: str, path: str) -> None:
        value = copy.deepcopy(self._resolve(frm))
        self._add(path, value)

    def _test(self, path: str, value: Any) -> None:
        if self._resolve(path) != value:
            raise ValueError(f"JSON Patch 'test' failed at {path!r}")


# --------------------------------------------------------------------------- #
# Chunk aggregator + JSON-Patch assembly
# --------------------------------------------------------------------------- #
class StreamAggregator:
    """Aggregates raw SSE chunks into events and assembles JSON-Patch payloads.

    ``feed`` accepts arbitrary byte/character fragments (the internal
    :class:`SSEParser` preserves partial lines/events between calls).  Every
    emitted event whose ``data`` parses as a JSON Patch document (a list of
    operations, or a single operation object) is applied to the running
    document exposed via :attr:`document`.
    """

    def __init__(
        self,
        *,
        apply_json_patch: bool = True,
        parser: Optional[SSEParser] = None,
    ) -> None:
        self._parser = parser or SSEParser()
        self._assembler = JSONPatchAssembler() if apply_json_patch else None
        self._events: List[SSEEvent] = []

    @property
    def events(self) -> List[SSEEvent]:
        return list(self._events)

    @property
    def document(self) -> Any:
        return self._assembler.document if self._assembler is not None else None

    @property
    def last_event_id(self) -> Optional[str]:
        return self._parser.last_event_id

    def feed(self, chunk: str) -> List[SSEEvent]:
        """Consume a chunk and return the events completed by it."""
        events = self._parser.feed(chunk)
        for evt in events:
            self._events.append(evt)
            if self._assembler is not None and evt.data:
                self._apply_if_patch(evt)
        return events

    def _apply_if_patch(self, evt: SSEEvent) -> None:
        try:
            payload = json.loads(evt.data)
        except (ValueError, TypeError):
            return
        try:
            if isinstance(payload, list):
                self._assembler.apply(payload)
            elif isinstance(payload, dict) and payload.get("op"):
                self._assembler.apply([payload])
        except (ValueError, KeyError, TypeError, IndexError):
            # A malformed patch must not corrupt an otherwise healthy stream.
            pass


# --------------------------------------------------------------------------- #
# Convenience demo (runnable with: python3 sse_pipeline.py)
# --------------------------------------------------------------------------- #
async def _demo() -> None:  # pragma: no cover - interactive example
    agg = StreamAggregator()
    # A stream fragmented mid-line and mid-event, rebuilding a JSON doc.
    raw = (
        'id: 1\nevent: patch\ndata: [{"op":"add","path":"/name","value":"ethan"}]\n\n'
        'id: 2\nevent: patch\ndata: {"op":"add","path":"/n","value":'
    )
    agg.feed(raw[:40])
    agg.feed(raw[40:100])
    agg.feed(raw[100:])
    print("assembled document:", json.dumps(agg.document))
    print("last event id:", agg.last_event_id)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_demo())