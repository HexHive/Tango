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
            self._listeners = {}

    def _notify_listeners(self, argt, ret):
        for loop, contexts in self._listeners.items():
            for context in contexts:
                run_coroutine_threadsafe(context.notify(argt, ret), loop)

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

    def _create_event_context(self):
        return EventContextManager(self)

    async def _listener_internal(self, cb, period):
        # FIXME access to self._listeners is not thread-safe; this could be an
        # issue when multiple WebUIs are being accessed for the same fuzzer.
        ctx = self._create_event_context()
        while not profiler.ProfilingStoppedEvent.is_set():
            async with ctx:
                if self._args is None:
                    error("Event triggered while args is None. Check for data races in code!")
                    continue
                current_task().last_visit = now()
                await cb(*self._args[0], **self._args[1], ret=self._ret)

            # ensure that we don't sleep needlessly between periods
            time_elapsed = now() - current_task().last_visit
            if period is not None and time_elapsed < period:
                await asyncio.sleep(period - time_elapsed)

    def listener(self, period=None):
        return partial(self._listener_internal, period=period)

class EventContextManager:
    def __init__(self, event_profiler):
        self._prof = event_profiler
        self._event = asyncio.Event()
        self.loop = get_running_loop()
        contexts = self._prof._listeners.get(self.loop, set())
        contexts.add(self)
        self._prof._listeners[self.loop] = contexts

    async def __aenter__(self):
        await self._event.wait()

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
        # consume the event at exit
        self._prof._args = None
        self._prof._ret = None
        self._event.clear()

    async def notify(self, args, ret):
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
        self._prof._args = args
        self._prof._ret = ret
        self._event.set()

    def __del__(self):
        if self._event is not None:
            self._prof._listeners.discard(self)
            self._event = None