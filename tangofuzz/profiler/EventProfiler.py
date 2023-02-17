from . import error

import profiler
from profiler import AbstractProfiler
from functools import partial
import asyncio
from asyncio import get_running_loop, run_coroutine_threadsafe
from time import perf_counter as now
from asyncio import iscoroutinefunction
from functools import wraps

class EventProfiler(AbstractProfiler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._args = None
        self._listeners = {}

    def _notify_listeners(self, argt, ret):
        for loop, contexts in self._listeners.items():
            for context in contexts:
                if not context.pending:
                    run_coroutine_threadsafe(context.notify(argt, ret), loop)

    def __call__(self, obj):
        @wraps(obj)
        def func(*args, **kwargs):
            ret = obj(*args, **kwargs)
            argt = (args, kwargs)
            self._notify_listeners(argt, ret)
            return ret
        @wraps(obj)
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

    def _create_event_context(self):
        ctx = EventContextManager(self)
        loop = get_running_loop()
        if (listeners := self._listeners.get(loop)) is not None:
            listeners.add(ctx)
        else:
            self._listeners[loop] = {ctx}
        return ctx

    async def _listener_internal(self, cb, period):
        # FIXME access to self._listeners is not thread-safe; this could be an
        # issue when multiple WebUIs are being accessed for the same fuzzer.
        ctx = self._create_event_context()
        last_visit = now()
        try:
            while True:
                async with ctx as profiling_stopped:
                    if profiling_stopped:
                        return
                    if ctx.args is None:
                        error("Event triggered while args is None. Check for data races in code!")
                        continue
                    await cb(*ctx.args[0], **ctx.args[1], ret=ctx.ret)

                    # ensure that we don't sleep needlessly between periods
                    time_elapsed = now() - last_visit
                    if period is not None and time_elapsed < period:
                        await asyncio.sleep(period - time_elapsed)
                last_visit = now()
        except asyncio.CancelledError:
            return

    def listener(self, period=None):
        return partial(self._listener_internal, period=period)

class EventContextManager:
    def __init__(self, event_profiler):
        self._prof = event_profiler
        self._event = asyncio.Event()
        self._stop_watcher = asyncio.ensure_future(profiler.ProfilingStoppedEvent.wait())

    async def __aenter__(self) -> bool:
        event_notify = asyncio.ensure_future(self._event.wait())
        done, _ = await asyncio.wait((self._stop_watcher, event_notify),
            return_when=asyncio.FIRST_COMPLETED)
        if self._stop_watcher in done:
            event_notify.cancel()
            return True
        else:
            return False

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
        # consume the event at exit
        self._args = None
        self._ret = None
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
        self._args = args
        self._ret = ret
        self._event.set()

    @property
    def pending(self) -> bool:
        return self._event.is_set()

    @property
    def args(self):
        return self._args

    @property
    def ret(self):
        return self._ret

    def __del__(self):
        self._stop_watcher.cancel()