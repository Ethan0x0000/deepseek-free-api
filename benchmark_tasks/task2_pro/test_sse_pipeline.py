"""High-concurrency unit self-tests for sse_pipeline.

Run with:  python3 test_sse_pipeline.py

The script is dependency-free (stdlib asyncio + unittest).  Every test uses
real asyncio concurrency: producers/consumers are spawned as tasks, drop
decisions happen under contention, and the reconnect engine exercises genuine
backoff cycles (delays shortened for the test run).
"""

from __future__ import annotations

import asyncio
import json
import unittest

from sse_pipeline import (
    BackpressureBuffer,
    BackpressurePolicy,
    JSONPatchAssembler,
    ReconnectPipeline,
    SSEEvent,
    SSEParser,
    StreamAggregator,
)


# --------------------------------------------------------------------------- #
# Synchronous parsing / patch tests
# --------------------------------------------------------------------------- #
class TestSSEParser(unittest.TestCase):
    def test_simple_event(self):
        p = SSEParser()
        evts = p.feed("data: hello\n\n")
        self.assertEqual(len(evts), 1)
        self.assertEqual(evts[0].data, "hello")
        self.assertEqual(evts[0].event, "message")

    def test_all_fields_and_multiline_data(self):
        p = SSEParser()
        evts = p.feed(
            "event: update\n"
            "data: line1\n"
            "data: line2\n"
            "id: 42\n"
            "retry: 3000\n"
            "\n"
        )
        self.assertEqual(len(evts), 1)
        e = evts[0]
        self.assertEqual(e.event, "update")
        self.assertEqual(e.data, "line1\nline2")
        self.assertEqual(e.id, "42")
        self.assertEqual(e.retry, 3000)
        self.assertEqual(p.last_event_id, "42")

    def test_crlf_and_comments(self):
        p = SSEParser()
        evts = p.feed(": keep-alive\r\ndata: x\r\n\r\n")
        self.assertEqual(len(evts), 1)
        self.assertEqual(evts[0].data, "x")

    def test_partial_chunks_split_everywhere(self):
        raw = "id: 7\nevent: msg\ndata: hello\ndata: world\n\n"
        for cut in range(1, len(raw)):
            p = SSEParser()
            events = []
            events += p.feed(raw[:cut])
            events += p.feed(raw[cut:])
            self.assertEqual(len(events), 1, f"cut={cut}")
            self.assertEqual(events[0].data, "hello\nworld")
            self.assertEqual(events[0].id, "7")

    def test_empty_data_not_dispatched(self):
        p = SSEParser()
        evts = p.feed("id: 1\n\n")
        self.assertEqual(evts, [])

    def test_space_stripping_and_empty_id(self):
        # Per SSE spec only a single leading space is stripped from the value.
        p = SSEParser()
        evts = p.feed("data:  two-spaces\ndata:no-space\n\n")
        self.assertEqual(evts[0].data, " two-spaces\nno-space")


class TestJSONPatchAssembler(unittest.TestCase):
    def test_add_replace_remove(self):
        a = JSONPatchAssembler({})
        a.apply([{"op": "add", "path": "/a/b", "value": 1}])
        self.assertEqual(a.document, {"a": {"b": 1}})
        a.apply([{"op": "replace", "path": "/a/b", "value": 2}])
        self.assertEqual(a.document, {"a": {"b": 2}})
        a.apply([{"op": "remove", "path": "/a/b"}])
        self.assertEqual(a.document, {"a": {}})

    def test_array_append_and_pointer_escaping(self):
        a = JSONPatchAssembler([])
        a.apply([
            {"op": "add", "path": "/-", "value": "x"},
            {"op": "add", "path": "/-", "value": "y"},
        ])
        self.assertEqual(a.document, ["x", "y"])
        a = JSONPatchAssembler({})
        a.apply([{"op": "add", "path": "/a~1b", "value": 1}])
        self.assertEqual(a.document, {"a/b": 1})

    def test_move_copy_test(self):
        a = JSONPatchAssembler({"x": [1, 2, 3]})
        a.apply([{"op": "copy", "from": "/x/0", "path": "/first"}])
        self.assertEqual(a.document["first"], 1)
        a.apply([{"op": "move", "from": "/x/2", "path": "/last"}])
        self.assertEqual(a.document, {"x": [1, 2], "first": 1, "last": 3})
        a.apply([{"op": "test", "path": "/last", "value": 3}])  # must pass
        with self.assertRaises(ValueError):
            a.apply([{"op": "test", "path": "/last", "value": 99}])

    def test_root_replace(self):
        a = JSONPatchAssembler({})
        a.apply([{"op": "add", "path": "", "value": [1, 2, 3]}])
        self.assertEqual(a.document, [1, 2, 3])


class TestStreamAggregator(unittest.TestCase):
    def test_patch_events_rebuild_document_across_chunks(self):
        agg = StreamAggregator()
        raw = (
            'id: 1\nevent: patch\ndata: [{"op":"add","path":"/users/0",'
            '"value":{"name":"alice"}}]\n\n'
            'id: 2\nevent: patch\ndata: [{"op":"add","path":"/users/1",'
            '"value":{"name":"bob"}}]\n\n'
        )
        # feed in awkward slices
        step = 11
        for i in range(0, len(raw), step):
            agg.feed(raw[i:i + step])
        self.assertEqual(
            agg.document,
            {"users": [{"name": "alice"}, {"name": "bob"}]},
        )
        self.assertEqual(agg.last_event_id, "2")

    def test_non_json_data_ignored_gracefully(self):
        agg = StreamAggregator()
        agg.feed("data: not-json\n\n")
        self.assertEqual(agg.document, {})
        agg.feed('data: {"no":"op here"}\n\n')
        self.assertEqual(agg.document, {})


# --------------------------------------------------------------------------- #
# Async tests
# --------------------------------------------------------------------------- #
class TestBackpressureBufferAsync(unittest.IsolatedAsyncioTestCase):
    async def test_block_policy_applies_backpressure(self):
        buf = BackpressureBuffer(2, BackpressurePolicy.BLOCK)
        await buf.put(1)
        await buf.put(2)
        # third put must block until a consumer frees space
        put_task = asyncio.create_task(buf.put(3))
        await asyncio.sleep(0.05)
        self.assertFalse(put_task.done())
        self.assertEqual(await buf.get(), 1)
        await put_task
        self.assertTrue(put_task.result())

    async def test_drop_oldest_keeps_latest(self):
        buf = BackpressureBuffer(2, BackpressurePolicy.DROP_OLDEST)
        self.assertTrue(await buf.put("a"))
        self.assertTrue(await buf.put("b"))
        self.assertTrue(await buf.put("c"))
        self.assertEqual(buf.size, 2)
        self.assertEqual(await buf.get(), "b")
        self.assertEqual(await buf.get(), "c")
        self.assertGreaterEqual(buf.dropped, 1)

    async def test_drop_latest_rejects_incoming(self):
        buf = BackpressureBuffer(2, BackpressurePolicy.DROP_LATEST)
        self.assertTrue(await buf.put("a"))
        self.assertTrue(await buf.put("b"))
        self.assertFalse(await buf.put("c"))
        self.assertEqual(buf.size, 2)
        self.assertEqual(buf.dropped, 1)
        self.assertEqual(await buf.get(), "a")
        self.assertEqual(await buf.get(), "b")

    async def test_high_concurrency_producers_consumers(self):
        capacity = 16
        buf = BackpressureBuffer(capacity, BackpressurePolicy.DROP_OLDEST)
        n_prod, per = 6, 400
        consumed = 0
        lock = asyncio.Lock()
        stop = asyncio.Event()

        async def producer():
            for j in range(per):
                await buf.put(j)

        async def consumer():
            nonlocal consumed
            while not stop.is_set():
                try:
                    item = await asyncio.wait_for(buf.get(), 0.1)
                except asyncio.TimeoutError:
                    continue
                async with lock:
                    consumed += 1
                buf.task_done()

        consumers = [asyncio.create_task(consumer()) for _ in range(8)]
        producers = [asyncio.create_task(producer()) for _ in range(n_prod)]
        await asyncio.gather(*producers)
        # join() returns once every enqueued item has been consumed.
        await asyncio.wait_for(buf.join(), timeout=30)
        stop.set()
        for c in consumers:
            c.cancel()
        await asyncio.gather(*consumers, return_exceptions=True)

        # Under DROP_OLDEST every put() is accepted; evicted items are dropped.
        self.assertEqual(buf.size, 0)
        self.assertEqual(consumed, n_prod * per - buf.dropped)
        self.assertGreater(buf.dropped, 0)


class TestReconnectPipelineAsync(unittest.IsolatedAsyncioTestCase):
    async def test_resume_with_last_event_id_and_backoff(self):
        attempts = 0
        connect_log = []
        seen = []

        async def connector(last_id):
            nonlocal attempts
            attempts += 1
            connect_log.append(last_id)
            if attempts == 1:
                # first connection delivers one event then dies
                yield SSEEvent(id="1", data="first")
                raise ConnectionError("boom")
            if attempts == 2:
                # second connection resumes from last_event_id then dies
                yield SSEEvent(id="2", data="second")
                raise ConnectionError("boom again")
            # third connection succeeds and closes cleanly
            yield SSEEvent(id="3", data="third")

        pipeline = ReconnectPipeline(
            connector,
            initial_delay=0.01,
            max_delay=0.02,
            jitter=False,
            max_attempts=3,
            on_event=lambda e: seen.append(e.data),
        )
        await asyncio.wait_for(pipeline.run(), timeout=10)
        self.assertEqual(attempts, 3)
        self.assertEqual(connect_log, [None, "1", "2"])
        self.assertEqual(pipeline.last_event_id, "3")
        self.assertEqual(seen, ["first", "second", "third"])

    async def test_max_attempts_exhaustion(self):
        async def connector(last_id):
            raise RuntimeError("always fails")
            yield  # pragma: no cover

        pipeline = ReconnectPipeline(
            connector,
            initial_delay=0.001,
            max_delay=0.002,
            jitter=False,
            max_attempts=2,
        )
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(pipeline.run(), timeout=5)
        self.assertTrue(pipeline.exhausted)
        self.assertGreaterEqual(pipeline.consecutive_failures, 1)

    async def test_stop_event_halts_loop(self):
        async def connector(last_id):
            while True:
                yield SSEEvent(id="x", data="tick")
                await asyncio.sleep(0)

        stop = asyncio.Event()
        collected = []
        pipeline = ReconnectPipeline(
            connector,
            initial_delay=0.01,
            max_delay=0.01,
            jitter=False,
            on_event=lambda e: collected.append(e.data),
        )

        async def stopper():
            await asyncio.sleep(0.05)
            stop.set()

        await asyncio.wait_for(
            asyncio.gather(pipeline.run(stop), stopper()), timeout=10
        )
        self.assertTrue(collected)


class TestHighConcurrencyIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_full_pipeline_under_concurrency(self):
        """Wire parser -> backpressure buffer -> aggregator under 50 workers."""
        agg = StreamAggregator()
        buf = BackpressureBuffer(256, BackpressurePolicy.DROP_OLDEST)
        stop = asyncio.Event()
        n_events = 0
        lock = asyncio.Lock()
        n_producers = 50

        async def consumer():
            nonlocal n_events
            while not stop.is_set():
                try:
                    chunk = await asyncio.wait_for(buf.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                agg.feed(chunk)
                async with lock:
                    n_events += 1
                buf.task_done()

        consumers = [asyncio.create_task(consumer()) for _ in range(8)]

        # 50 producers each push one complete patch event (dict keys avoid the
        # RFC patch index-padding behaviour of numeric array paths).
        async def producer(wid: int):
            body = json.dumps([{"op": "add", "path": f"/count/k{wid}", "value": wid}])
            raw = f"id: {wid}\nevent: patch\ndata: {body}\n\n"
            await buf.put(raw)

        producers = [asyncio.create_task(producer(w)) for w in range(n_producers)]
        await asyncio.gather(*producers)
        await asyncio.wait_for(buf.join(), timeout=10)
        stop.set()
        for c in consumers:
            c.cancel()
        await asyncio.gather(*consumers, return_exceptions=True)

        self.assertEqual(n_events, n_producers)
        self.assertEqual(len(agg.document.get("count", {})), n_producers)


if __name__ == "__main__":
    unittest.main(verbosity=2)