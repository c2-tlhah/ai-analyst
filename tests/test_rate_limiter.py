import pytest

from app.llm.rate_limiter import RateLimitQueueTimeout, SlidingWindowRateLimiter


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_limiter_spaces_calls_and_enforces_rolling_window():
    fake = FakeTime()
    limiter = SlidingWindowRateLimiter(
        limit=3,
        window_seconds=6,
        min_interval_seconds=1,
        max_wait_seconds=20,
        clock=fake.clock,
        sleep=fake.sleep,
    )

    first = limiter.acquire()
    second = limiter.acquire()
    third = limiter.acquire()
    fourth = limiter.acquire()

    assert first.waited_seconds == 0
    assert second.waited_seconds == pytest.approx(1)
    assert third.waited_seconds == pytest.approx(1)
    assert fourth.waited_seconds == pytest.approx(4)
    assert fake.now == pytest.approx(6)
    assert fourth.requests_in_window <= 3


def test_429_penalty_is_shared_with_the_next_request():
    fake = FakeTime()
    limiter = SlidingWindowRateLimiter(
        limit=60,
        min_interval_seconds=0,
        max_wait_seconds=70,
        clock=fake.clock,
        sleep=fake.sleep,
    )
    limiter.acquire()
    limiter.penalize(60)

    permit = limiter.acquire()

    assert permit.waited_seconds == pytest.approx(60)
    assert fake.now == pytest.approx(60)


def test_queue_aborts_when_wait_would_exceed_safety_budget():
    fake = FakeTime()
    limiter = SlidingWindowRateLimiter(
        limit=1,
        window_seconds=60,
        min_interval_seconds=0,
        max_wait_seconds=5,
        clock=fake.clock,
        sleep=fake.sleep,
    )
    limiter.acquire()

    with pytest.raises(RateLimitQueueTimeout, match="exceeding"):
        limiter.acquire()
