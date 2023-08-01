from __future__ import annotations

from . import error

from tango.common import get_session_context

from sys import gettrace as sys_gettrace
from abc import ABC, ABCMeta, abstractmethod
from math import trunc
from functools import partial, wraps, lru_cache
from time import perf_counter as timestamp
from collections import deque
from statistics import mean
from datetime import datetime, timedelta
from contextvars import ContextVar
from typing import Optional, Iterable, Any
import asyncio
import os

__all__ = [
    'initialize', 'get_profiler', 'get_all_profilers', 'is_profiling_active',
    'AbstractProfiler', 'ValueProfiler', 'LambdaProfiler', 'NumericalProfiler',
    'FunctionCallProfiler', 'EventProfiler', 'PeriodicProfiler',
    'FrequencyProfiler', 'CountProfiler', 'ValueMeanProfiler',
    'LambdaMeanProfiler', 'TimeElapsedProfiler', 'AbstractProfilerMeta'
]

ProfiledObjects = {}
ProfilingTasks = []

EnabledProfilers = set(
    name.split('TANGO_PROFILE_', 1)[1] for name in os.environ.keys()
    if name.startswith('TANGO_PROFILE_'))

DefaultProfilers = {}
DefaultProfilers['minimal'] = \
    {'time_elapsed', 'resets', 'instructions', 'execs', 'gens',
     'total_instructions', 'snapshots', 'crash'}
DefaultProfilers['web'] = DefaultProfilers['minimal'] | \
    {'webui', 'perform_instruction', 'target_name', 'status'}

async def initialize(tg: Optional[asyncio.TaskGroup]=None):
    for idx, coro in enumerate(ProfilingTasks):
        factory = tg or asyncio
        ProfilingTasks[idx] = factory.create_task(coro)

def get_profiler(name: str, *default: Any) -> AbstractProfiler:
    assert len(default) <= 1
    if default:
        p = ProfiledObjects.get(name, *default)
    else:
        p = ProfiledObjects[name]
    if isinstance(p, ContextVar):
        context = get_session_context()
        if p not in context and default:
            return default[0]
        return context[p]
    else:
        return p

def get_all_profilers() -> Iterable[tuple[str, AbstractProfiler]]:
    return {name: profiler for name in ProfiledObjects
            if (profiler := get_profiler(name, None))}.items()

@lru_cache(maxsize=32)
def is_profiling_active(*names: Iterable[str]) -> bool:
    if not AbstractProfilerMeta.ProfilingNOP:
        return True
    active = True
    for name in names:
        active = active and (ProfiledObjects.get(name) or \
            name in EnabledProfilers or \
            any(name in DefaultProfilers.get(profset, ())
                for profset in EnabledProfilers))
    return active

class AbstractProfilerMeta(ABCMeta):
    DEBUG = sys_gettrace() is not None
    ProfilingNOP = DEBUG or os.environ.get('TANGO_NO_PROFILE')

    def __call__(cls, name, /, *args,
            session_local: bool=True, **kwargs):
        if not is_profiling_active(name):
            return cls.nop
        if (var := ProfiledObjects.get(name)) is None or \
                (session_local and not var in (context := get_session_context())):
            p = super().__call__(*args, **kwargs)
            object.__setattr__(p, 'name', name)
            if session_local:
                context = get_session_context()
                if not var:
                    var = ProfiledObjects[name] = ContextVar(f'profiler[{name}]')
                try:
                    context.run(var.set, p)
                except RuntimeError:
                    var.set(p)
            else:
                ProfiledObjects[name] = p
        elif session_local:
            p = context[var]
        else:
            p = var
        return p

    @staticmethod
    def nop(obj=None):
        return obj

class AbstractProfiler(ABC, metaclass=AbstractProfilerMeta):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, obj):
        return obj

    def __str__(self):
        raise NotImplementedError

class ValueProfiler(AbstractProfiler):
    def __call__(self, obj):
        obj = self._value = super().__call__(obj)
        return obj

    @property
    def value(self):
        return self._value

    def __str__(self):
        return str(self.value)

class LambdaProfiler(ValueProfiler):
    @property
    def value(self):
        return self._value()

class NumericalProfiler(AbstractProfiler):
    def __init__(self, *, decimal_digits=1, **kwargs):
        super().__init__(**kwargs)
        self._step = 10 ** decimal_digits

    @property
    @abstractmethod
    def value(self):
        pass

    def truncate(self, value):
        return trunc(value * self._step) / self._step

    def __str__(self):
        return str(self.truncate(self.value))

class LocalGlobalProfilerMeta(AbstractProfilerMeta):
    def __call__(cls, name, /, *args, **kwargs):
        kwargs['session_local'] = False
        return super().__call__(name, *args, **kwargs)

    # def __new__(metacls, name, bases, namespace):
    #     if not namespace.get('__init_exempt__') and namespace.pop('__init__', None):
    #         raise TypeError("Subclasses cannot implement an __init__!")
    #     return super().__new__(metacls, name, bases, namespace)

class LocalGlobalProfiler(AbstractProfiler, metaclass=LocalGlobalProfilerMeta):
    # __init_exempt__ = True

    # def __init__(self, *, name: str, session_local_attrs: bool=True, **kwargs):
    #     # we use super() to avoid invoking our own setattr here
    #     super().__setattr__('_session_local', session_local_attrs)
    #     if session_local_attrs:
    #         var = ContextVar(f'{name}._attr_storage')
    #         super().__setattr__('_attr_storage', var)
    #     super().__init__(**kwargs)

    # def __getattr__(self, name):
    #     # not circular logic, since __getattribute__ will find it in self
    #     if self._session_local:
    #         try:
    #             context = get_session_context()
    #             return context[self._attr_storage][name]
    #         except KeyError as ex:
    #             raise AttributeError() from ex
    #     else:
    #         # __setattr__ uses the original name, so that
    #         # super().__getattribute___(name) should actually find it;
    #         # if it doesn't, then it was never set
    #         raise AttributeError

    # def __setattr__(self, name, value):
    #     if self._session_local:
    #         context = get_session_context()
    #         if (stor := context.get(self._attr_storage)) is None:
    #             stor = context[self._attr_storage] = {}
    #         stor[name] = value
    #     else:
    #         super().__setattr__(name, value)
    pass

class FunctionCallProfiler(LocalGlobalProfiler):
    def __call__(self, fn):
        fn = super().__call__(fn)
        @wraps(fn)
        def func(*args, **kwargs):
            ret = fn(*args, **kwargs)
            self.callback(ret, *args, **kwargs)
            return ret
        @wraps(fn)
        async def afunc(*args, **kwargs):
            ret = await fn(*args, **kwargs)
            self.callback(ret, *args, **kwargs)
            return ret

        if asyncio.iscoroutinefunction(fn):
            wrapper = afunc
        else:
            wrapper = func
        wrapper = wraps(fn)(wrapper)
        return wrapper

    @abstractmethod
    def callback(self, returncode, /, *args, **kwargs):
        pass

class EventProfiler(FunctionCallProfiler):
    def _check_init_attrs(self):
        if not hasattr(self, '_listeners'):
            self._listeners = {}

    def _notify_listeners(self, argt, ret):
        self._check_init_attrs()
        for loop, contexts in self._listeners.items():
            for context in contexts:
                if not context.pending:
                    asyncio.run_coroutine_threadsafe(context.notify(argt, ret),
                        loop=loop)

    def callback(self, returncode, /, *args, **kwargs):
        argt = (args, kwargs)
        self._notify_listeners(argt, returncode)

    def _create_event_context(self):
        ctx = EventContextManager(self)
        loop = asyncio.get_running_loop()
        if (listeners := self._listeners.get(loop)) is not None:
            listeners.add(ctx)
        else:
            self._listeners[loop] = {ctx}
        return ctx

    async def _listener_internal(self, cb, period):
        ctx = self._create_event_context()
        last_visit = timestamp()
        try:
            while True:
                async with ctx:
                    if ctx.args is None:
                        error("Event triggered while args is None. Check for data races in code!")
                        continue
                    await cb(*ctx.args[0], **ctx.args[1], ret=ctx.ret)

                    # ensure that we don't sleep needlessly between periods
                    time_elapsed = timestamp() - last_visit
                    if period is not None and time_elapsed < period:
                        await asyncio.sleep(period - time_elapsed)
                last_visit = timestamp()
        except asyncio.CancelledError:
            return

    def listener(self, period=None):
        self._check_init_attrs()
        return partial(self._listener_internal, period=period)

class EventContextManager:
    def __init__(self, event_profiler):
        self._prof = event_profiler
        self._event = asyncio.Event()

    async def __aenter__(self) -> bool:
        await self._event.wait()

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

class PeriodicProfiler(AbstractProfiler):
    def __init__(self, *, period=1, **kwargs):
        super().__init__(**kwargs)
        self._period = period
        ProfilingTasks.append(self._task_worker())

    async def _task_worker(self):
        try:
            while True:
                await asyncio.sleep(self._period)
                self.do_task()
        except asyncio.CancelledError:
            return

    @abstractmethod
    def do_task(self):
        pass

class FrequencyProfiler(FunctionCallProfiler, PeriodicProfiler, NumericalProfiler):
    def _check_init_attrs(self):
        if not hasattr(self, '_counter'):
            self._counter = 0
        if not hasattr(self, '_frequency'):
            self._frequency = None

    def callback(self, returncode, /, *args, **kwargs):
        self._check_init_attrs()
        self._counter += 1

    def do_task(self):
        self._check_init_attrs()
        self._frequency = self._counter / self._period
        self._counter = 0

    @property
    def value(self) -> float:
        return self._frequency

class CountProfiler(ValueProfiler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._count = 0

    def __call__(self, num):
        num = super().__call__(num)
        self._count += num
        return num

    @staticmethod
    def _format(num):
        for unit in ['','K']:
            if abs(num) < 1000.0:
                return "%.1f%s" % (num, unit)
            num /= 1000.0
        return "%.1f%s" % (num, 'M')

    def __str__(self):
        return self._format(self.value)

    @property
    def value(self) -> int:
        return self._count

class ValueMeanProfiler(NumericalProfiler):
    def __init__(self, *, samples=10, **kwargs):
        super().__init__(**kwargs)
        self._maxlen = samples
        if self._maxlen > 0:
            self._samples = deque(maxlen=self._maxlen)
            self.___call_internal___ = self.___call_lastn___
            self._calculate = lambda: mean(self._samples) if self._samples else None
        else:
            self._count = 0
            self._sum = 0
            self.___call_internal___ = self.___call_total___
            self._calculate = lambda: self._sum / self._count if self._count != 0 else None

    def __call__(self, obj):
        obj = super().__call__(obj)
        return self.___call_internal___(obj)

    def ___call_total___(self, obj):
        self._sum += obj
        self._count += 1
        return obj

    def ___call_lastn___(self, obj):
        self._samples.append(obj)
        return obj

    @property
    def value(self) -> float:
        return self._calculate()

class LambdaMeanProfiler(PeriodicProfiler, NumericalProfiler):
    def __init__(self, *, samples=10, **kwargs):
        super().__init__(**kwargs)
        self._samples = deque(maxlen=samples)
        self._mean = None

    def __call__(self, lmda):
        lmda = super().__call__(lmda)
        self._lmda = lmda
        return lmda

    def do_task(self):
        self._samples.append(self._lmda())
        self._mean = mean(self._samples)

    @property
    def value(self) -> float:
        return self._mean

class TimeElapsedProfiler(AbstractProfiler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._start = datetime.now()
        self._running = True
        self._accum = timedelta()

    def __call__(self):
        self.toggle()

    def toggle(self):
        now = datetime.now()
        if self._running:
            self._accum += now - self._start
            self._start = now
            self._running = False
        else:
            self._start = now
            self._running = True

    def __str__(self):
        return str(self.timedelta).split('.')[0]

    @property
    def value(self) -> float:
        return self.timedelta.total_seconds()

    @property
    def timedelta(self) -> timedelta:
        delta = self._accum
        if self._running:
            delta += datetime.now() - self._start
        return delta