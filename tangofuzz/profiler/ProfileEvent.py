from . import error

from profiler import ProfilerBase, ProfilingStoppedEvent
from functools import partial
import asyncio
from asyncio import (get_running_loop, run_coroutine_threadsafe, wait_for,
                    current_task)
from time import perf_counter as now

class ProfileEvent(ProfilerBase):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        if not self._init_called:
            self._args = None
            self._listeners = []

    def __call__(self, obj):
        def func(*args, **kwargs):
            ret = obj(*args, **kwargs)
            self._args = (args, kwargs)
            self._ret = ret
            for loop in self._listeners:
                run_coroutine_threadsafe(self._listener_notify(), loop)
            return ret
        return func

    @property
    def value(self):
        raise NotImplemented()

    @property
    def args(self):
        return self._args

    @property
    def ret(self):
        return self._ret

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
        self._args = None
        self._ret = None
        self._listener_event.clear()

    @property
    def _listener_event(self):
        return get_running_loop().events[id(self)]

    @_listener_event.setter
    def _listener_event(self, event):
        get_running_loop().events[id(self)] = event

    async def _listener_notify(self):
        self._listener_event.set()

    async def _listener_internal(self, cb, period):
        # FIXME access to self._listeners is not thread-safe; this could be an
        # issue when multiple WebUIs are being accessed for the same fuzzer.
        try:
            self._listeners.append(get_running_loop())
            self._listener_event = asyncio.Event()
            while not ProfilingStoppedEvent.is_set():
                while self._args is None:
                    await self._listener_event.wait()
                async with self:
                    current_task().last_visit = now()
                    await cb(*self._args[0], **self._args[1], ret=self._ret)

                # ensure that we don't sleep needlessly between periods
                time_elapsed = now() - current_task().last_visit
                if time_elapsed < period:
                    await asyncio.sleep(period - time_elapsed)
        except Exception as ex:
            import traceback
            error(f'{traceback.format_exc()}')
        finally:
            del get_running_loop().events[id(self)]
            self._listeners.remove(get_running_loop())

    def listener(self, period=None):
        return partial(self._listener_internal, period=period)