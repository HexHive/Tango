from . import error

import profiler
from profiler import ProfilerBase
from functools import partial
import asyncio
from asyncio import (get_running_loop, run_coroutine_threadsafe, wait_for,
                    current_task)
from time import perf_counter as now
from asyncio import iscoroutinefunction

class ProfileEvent(ProfilerBase):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        if not self._init_called:
            self._args = None
            self._listeners = []

    def _notify_listeners(self, argt, ret):
        for loop in self._listeners:
            run_coroutine_threadsafe(self._listener_notify(argt, ret), loop)

    def ___call___(self, obj):
        def func(*args, **kwargs):
            ret = obj(*args, **kwargs)
            argt = (args, kwargs)
            self._notify_listeners(argt, ret)
            return ret
        async def afunc(*args, **kwargs):
            ret = await obj(*args, **kwargs)
            argt = (args, kwargs)
            self._notify_listeners(argt, ret)
            return ret

        if iscoroutinefunction(obj):
            return afunc
        else:
            return func

    @property
    def value(self):
        raise NotImplementedError()

    @property
    def args(self):
        return self._args

    @property
    def ret(self):
        return self._ret

    async def __aenter__(self):
        await self._listener_event.wait()
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

    async def _listener_notify(self, args, ret):
        # Since this is a coroutine, it could be scheduled at an arbitrary point
        # in time and could thus result in setting the event long after the args
        # had been consumed by the original function. To ensure that the args
        # are delivered properly to the callback, we set self._args immediately
        # before setting the event.
        #
        # Otherwise, if args were set in the func wrapper above, then the
        # coroutine is scheduled, the following could happen:
        # * self.__aenter()__ consumes the previous event
        # * func() sets args and schedules this coroutine
        # * self.__aexit()__ clears args
        # * this coroutine is executed and the event is set
        # * self.__aenter()__ consumes this event, but args are None
        self._args = args
        self._ret = ret
        self._listener_event.set()

    async def _listener_internal(self, cb, period):
        # FIXME access to self._listeners is not thread-safe; this could be an
        # issue when multiple WebUIs are being accessed for the same fuzzer.
        try:
            self._listeners.append(get_running_loop())
            self._listener_event = asyncio.Event()
            while not profiler.ProfilingStoppedEvent.is_set():
                async with self:
                    if self._args is None:
                        error("Event triggered while args is None. Check for data races in code!")
                        continue
                    current_task().last_visit = now()
                    await cb(*self._args[0], **self._args[1], ret=self._ret)

                # ensure that we don't sleep needlessly between periods
                time_elapsed = now() - current_task().last_visit
                if period is not None and time_elapsed < period:
                    await asyncio.sleep(period - time_elapsed)
        finally:
            del get_running_loop().events[id(self)]
            self._listeners.remove(get_running_loop())

    def listener(self, period=None):
        return partial(self._listener_internal, period=period)