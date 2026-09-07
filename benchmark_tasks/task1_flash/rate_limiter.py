import asyncio
import time
from collections import deque


class SlidingWindowRateLimiter:
    """Sliding window log rate limiter."""

    def __init__(self, limit: int, window_secs: float):
        self.limit = limit
        self.window_secs = window_secs
        self.timestamps = deque()
        self._lock = asyncio.Lock()

    async def _cleanup(self, now: float) -> None:
        while self.timestamps and now - self.timestamps[0] > self.window_secs:
            self.timestamps.popleft()

    async def acquire(self) -> bool:
        """Attempt to acquire a permit. Returns True if allowed."""
        async with self._lock:
            now = time.time()
            await self._cleanup(now)
            if len(self.timestamps) < self.limit:
                self.timestamps.append(now)
                return True
            return False

    async def __aenter__(self):
        if not await self.acquire():
            raise RuntimeError("Rate limit exceeded")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TokenBucketRateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def _refill(self, now: float) -> None:
        elapsed = now - self.last_refill
        if elapsed > 0:
            new_tokens = elapsed * self.rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now

    async def acquire(self) -> bool:
        """Attempt to consume one token. Returns True if successful."""
        async with self._lock:
            now = time.time()
            await self._refill(now)
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def __aenter__(self):
        if not await self.acquire():
            raise RuntimeError("Rate limit exceeded")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class LeakyBucketRateLimiter:
    """Leaky bucket rate limiter with a background drainer."""

    def __init__(self, capacity: int, leak_rate: float):
        self.capacity = capacity
        self.leak_rate = leak_rate  # items per second
        self.queue = asyncio.Queue(maxsize=capacity)
        self._lock = asyncio.Lock()
        self._running = True
        self._leak_task = asyncio.create_task(self._leak())

    async def _leak(self) -> None:
        """Background coroutine that drains the queue at a constant rate."""
        interval = 1.0 / self.leak_rate
        while self._running:
            await asyncio.sleep(interval)
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

    async def acquire(self) -> bool:
        """Attempt to add a request to the queue. Returns True if space available."""
        async with self._lock:
            if self.queue.qsize() < self.capacity:
                await self.queue.put(time.time())  # store timestamp (optional)
                return True
            return False

    async def __aenter__(self):
        if not await self.acquire():
            raise RuntimeError("Rate limit exceeded")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def close(self) -> None:
        """Stop the background leak task."""
        self._running = False
        self._leak_task.cancel()
