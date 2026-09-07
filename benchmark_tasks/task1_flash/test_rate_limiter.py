import asyncio
import time
from rate_limiter import (
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
    LeakyBucketRateLimiter,
)


async def test_sliding_window():
    """Test sliding window limiter."""
    limiter = SlidingWindowRateLimiter(limit=3, window_secs=1.0)

    # Should allow 3 requests immediately
    results = []
    for _ in range(3):
        results.append(await limiter.acquire())
    assert all(results), "SlidingWindow: first 3 should be allowed"

    # 4th should be rejected
    assert not await limiter.acquire(), "SlidingWindow: 4th should be rejected"

    # Wait just over window, should allow again
    await asyncio.sleep(1.1)
    assert await limiter.acquire(), "SlidingWindow: after window, should allow"

    print("✅ SlidingWindowRateLimiter passed")


async def test_token_bucket():
    """Test token bucket limiter."""
    limiter = TokenBucketRateLimiter(rate=2.0, capacity=4)

    # Should allow up to capacity immediately
    for i in range(4):
        assert await limiter.acquire(), f"TokenBucket: request {i+1} should be allowed"
    # 5th should fail
    assert not await limiter.acquire(), "TokenBucket: 5th should be rejected"

    # Wait for tokens to regenerate (2 tokens per second -> 0.5s per token)
    await asyncio.sleep(0.6)
    assert await limiter.acquire(), "TokenBucket: after refill, should allow"

    print("✅ TokenBucketRateLimiter passed")


async def test_leaky_bucket():
    """Test leaky bucket limiter."""
    limiter = LeakyBucketRateLimiter(capacity=2, leak_rate=1.0)

    # Should allow up to capacity
    for i in range(2):
        assert await limiter.acquire(), f"LeakyBucket: request {i+1} should be allowed"
    # 3rd should fail
    assert not await limiter.acquire(), "LeakyBucket: 3rd should be rejected"

    # Wait for leak to drain one
    await asyncio.sleep(1.1)
    assert await limiter.acquire(), "LeakyBucket: after leak, should allow"

    limiter.close()
    print("✅ LeakyBucketRateLimiter passed")


async def test_context_manager_sliding():
    """Test SlidingWindowRateLimiter as context manager."""
    limiter = SlidingWindowRateLimiter(limit=2, window_secs=1.0)

    async with limiter:
        pass
    async with limiter:
        pass
    try:
        async with limiter:
            pass  # third should exceed
    except RuntimeError:
        pass
    else:
        raise AssertionError("Context manager should raise on over-limit")

    print("✅ SlidingWindowRateLimiter context manager passed")


async def test_context_manager_token():
    """Test TokenBucketRateLimiter as context manager."""
    limiter = TokenBucketRateLimiter(rate=10.0, capacity=1)

    async with limiter:
        pass
    try:
        async with limiter:
            pass  # should fail because capacity is 1 and rate is high but still need time
    except RuntimeError:
        pass
    else:
        raise AssertionError("Context manager should raise on over-limit")

    print("✅ TokenBucketRateLimiter context manager passed")


async def test_context_manager_leaky():
    """Test LeakyBucketRateLimiter as context manager."""
    limiter = LeakyBucketRateLimiter(capacity=1, leak_rate=10.0)

    async with limiter:
        pass
    try:
        async with limiter:
            pass  # should fail because queue is full
    except RuntimeError:
        pass
    else:
        raise AssertionError("Context manager should raise on over-limit")

    limiter.close()
    print("✅ LeakyBucketRateLimiter context manager passed")


async def high_concurrency_test():
    """High-concurrency test for all limiters."""

    # Sliding window: 100 requests, window=0.2s, limit=10 per window
    sliding = SlidingWindowRateLimiter(limit=10, window_secs=0.2)
    async def worker_sliding(i):
        return await sliding.acquire()
    tasks = [worker_sliding(i) for i in range(100)]
    results = await asyncio.gather(*tasks)
    allowed = sum(1 for r in results if r)
    # Within a short window, only ~10 should succeed (some may be just after window expires)
    assert 8 <= allowed <= 15, f"Sliding high-concurrency allowed {allowed}, expected ~10"
    print(f"✅ SlidingWindow high-concurrency passed (allowed {allowed})")

    # Token bucket: rate=50/s, capacity=20, 100 requests
    token = TokenBucketRateLimiter(rate=50, capacity=20)
    async def worker_token(i):
        return await token.acquire()
    tasks = [worker_token(i) for i in range(100)]
    results = await asyncio.gather(*tasks)
    allowed = sum(1 for r in results if r)
    # Should allow exactly capacity initially (20) plus refills during burst
    assert 20 <= allowed <= 30, f"Token high-concurrency allowed {allowed}, expected ~20-30"
    print(f"✅ TokenBucket high-concurrency passed (allowed {allowed})")

    # Leaky bucket: capacity=5, leak_rate=10/s, 50 requests
    leaky = LeakyBucketRateLimiter(capacity=5, leak_rate=10)
    async def worker_leaky(i):
        return await leaky.acquire()
    tasks = [worker_leaky(i) for i in range(50)]
    results = await asyncio.gather(*tasks)
    allowed = sum(1 for r in results if r)
    # Should allow up to capacity initially, plus some leaked
    assert 5 <= allowed <= 15, f"Leaky high-concurrency allowed {allowed}, expected ~5-15"
    leaky.close()
    print(f"✅ LeakyBucket high-concurrency passed (allowed {allowed})")


async def main():
    print("Running rate limiter tests...\n")
    await test_sliding_window()
    await test_token_bucket()
    await test_leaky_bucket()
    print()
    await test_context_manager_sliding()
    await test_context_manager_token()
    await test_context_manager_leaky()
    print()
    await high_concurrency_test()
    print("\n🎉 All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
