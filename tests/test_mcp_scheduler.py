"""Tests for Scheduler — registration, jitter, execution, error handling."""

from __future__ import annotations

import asyncio

import pytest

from mdrouter.mcp.framework.config import SchedulerConfig
from mdrouter.mcp.framework.scheduler import Scheduler


@pytest.fixture
def sched_config():
    return SchedulerConfig(enabled=True, default_interval_hours=1, jitter_seconds=0)


@pytest.fixture
def sched(sched_config):
    return Scheduler(sched_config)


async def test_scheduler_disabled():
    """When disabled, registering tasks should be a no-op."""
    config = SchedulerConfig(enabled=False, default_interval_hours=1)
    s = Scheduler(config)

    called = False

    async def _task():
        nonlocal called
        called = True

    s.register("test", _task, run_on_startup=True)
    await s.start()
    await asyncio.sleep(0.1)
    await s.stop()
    assert not called


async def test_register_with_startup(sched):
    """Task with run_on_startup should fire immediately."""
    called = False

    async def _task():
        nonlocal called
        called = True

    sched.register("test", _task, run_on_startup=True)
    await sched.start()
    await asyncio.sleep(0.2)
    await sched.stop()
    assert called


async def test_register_without_startup(sched):
    """Task without run_on_startup should not fire immediately."""
    called = False

    async def _task():
        nonlocal called
        called = True

    sched.register("test", _task, run_on_startup=False)
    await sched.start()
    await asyncio.sleep(0.2)
    await sched.stop()
    assert not called


async def test_error_handling(sched):
    """Failing task should not crash the scheduler."""
    call_count = 0

    async def _failing_task():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Simulated failure")

    sched.register("failing", _failing_task, run_on_startup=True, interval_hours=0)
    await sched.start()
    await asyncio.sleep(0.3)
    await sched.stop()

    # Should have been called at least once despite the error
    assert call_count >= 1


async def test_stop_cancels_tasks(sched):
    """After stop, tasks should not execute."""
    call_count = 0

    async def _task():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(10)

    sched.register("long", _task, run_on_startup=True, interval_hours=0)
    await sched.start()
    await asyncio.sleep(0.2)
    assert call_count >= 1

    await sched.stop()
    count_after_stop = call_count
    await asyncio.sleep(0.2)
    assert call_count == count_after_stop  # No more calls


async def test_task_count(sched):
    """Task count should reflect registered tasks."""
    assert sched.task_count == 0

    async def _t1():
        pass

    async def _t2():
        pass

    sched.register("t1", _t1)
    sched.register("t2", _t2)
    assert sched.task_count == 2
