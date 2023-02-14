from . import debug, info, warning

import asyncio
import functools
from async_property import async_property, async_cached_property
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import time

# This is a single-threaded executor to be used for wrapping sync methods into
# async coroutines. It is single-threaded because, for ptrace to work correctly,
# the tracer must be the same thread that launched the process.
# Since ProcessLoader uses sync_to_async for _launch_target, then the parent is
# this single thread in the thread pool, and all following calls to ptrace also
# use the same thread pool (see {TCP,UDP}Channel.{send,receive}).
GLOBAL_ASYNC_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='AsyncWrapper')

async def async_enumerate(asequence, start=0):
    """Asynchronously enumerate an async iterator from a given start value"""
    n = start
    async for elem in asequence:
        yield n, elem
        n += 1

class Suspendable:
    """
    Adapted from:
    https://stackoverflow.com/questions/66687549/is-it-possible-to-suspend-and-restart-tasks-in-async-python
    """
    def __init__(self, target, /, **kwargs):
        self._target = target
        self._init_kwargs = kwargs

    def _initialize(self, *, suspend_cb=None, resume_cb=None,
            sleep_task=None, wakeup_task=None):
        self._can_run = asyncio.Event()
        self._can_run.set()
        self._awaited = False
        self.set_callbacks(suspend_cb=suspend_cb, resume_cb=resume_cb)
        self.set_tasks(sleep_task=sleep_task, wakeup_task=wakeup_task)

        current_task = asyncio.current_task()
        if not hasattr(current_task, 'suspendable_ancestors'):
            current_task.suspendable_ancestors = []
            self._parent = self
        else:
            self._parent = current_task.suspendable_ancestors[-1]

        self.child_suspendables = []
        self.child_suspend_cbs = []
        self.child_resume_cbs = []
        self._push_suspend_idx = len(self._parent.child_suspend_cbs)
        self._push_resume_idx = len(self._parent.child_resume_cbs)

    def set_callbacks(self, **kwargs):
        def set_suspend_cb(cb):
            if self._awaited:
                if self._suspend_cb:
                    self._parent.child_suspend_cbs[self._push_suspend_idx] = cb
                else:
                    self._parent.child_suspend_cbs.insert(self._push_suspend_idx, cb)
            self._suspend_cb = cb
        def set_resume_cb(cb):
            if self._awaited:
                if self._resume_cb:
                    self._parent.child_resume_cbs[self._push_resume_idx] = cb
                else:
                    self._parent.child_resume_cbs.insert(self._push_resume_idx, cb)
            self._resume_cb = cb

        try:
            set_suspend_cb(kwargs['suspend_cb'])
        except KeyError:
            pass
        try:
            set_resume_cb(kwargs['resume_cb'])
        except KeyError:
            pass

    def set_tasks(self, **kwargs):
        def set_sleep_task(task):
            self._sleep_task = task
        def set_wakeup_task(task):
            self._wakeup_task = task

        try:
            set_sleep_task(kwargs['sleep_task'])
        except KeyError:
            pass
        try:
            set_wakeup_task(kwargs['wakeup_task'])
        except KeyError:
            pass

    @property
    def tasks(self):
        return self._sleep_task, self._wakeup_task

    @property
    def callbacks(self):
        return self._suspend_cb, self._resume_cb

    def push_cbs(self):
        asyncio.current_task().suspendable_ancestors.append(self)
        if self._suspend_cb:
            assert self._push_suspend_idx == len(self._parent.child_suspend_cbs)
            self._parent.child_suspend_cbs.append(self._suspend_cb)
        if self._resume_cb:
            assert self._push_resume_idx == len(self._parent.child_resume_cbs)
            self._parent.child_resume_cbs.append(self._resume_cb)
        if self._parent is not self:
            self._parent.child_suspendables.append(self)
        self._awaited = True

    def pop_cbs(self):
        self._awaited = False
        if self._suspend_cb:
            assert self._suspend_cb is self._parent.child_suspend_cbs.pop()
        if self._resume_cb:
            assert self._resume_cb is self._parent.child_resume_cbs.pop()
        if self._parent is not self:
            assert self is self._parent.child_suspendables.pop()
        assert asyncio.current_task().suspendable_ancestors.pop() == self

    def __await__(self):
        self._initialize(**self._init_kwargs)
        self.push_cbs()

        try:
            target_iter = self._target.__await__()
            iter_send, iter_throw = target_iter.send, target_iter.throw
            send, message = iter_send, None
            # This "while" emulates yield from.
            while True:
                # wait for can_run before resuming execution of self._target
                slept = False
                try:
                    while not self._can_run.is_set():
                        if not slept and self._sleep_task:
                            yield from self._sleep_task.__await__()
                            slept = True
                        yield from self._can_run.wait().__await__()
                        if self._can_run.is_set() and self._wakeup_task:
                            yield from self._wakeup_task.__await__()
                            break
                except BaseException as err:
                    send, message = iter_throw, err

                # continue with our regular program
                try:
                    signal = send(message)
                except StopIteration as err:
                    return err.value
                else:
                    send = iter_send
                try:
                    message = yield signal
                except BaseException as err:
                    send, message = iter_throw, err
        finally:
            self.pop_cbs()

    def suspend(self):
        for child in self.child_suspendables[::-1]:
            child.suspend()
        for cb in self.child_suspend_cbs[::-1]:
            cb()
        self._can_run.clear()
        warning(f"Suspended all children {self._target=}")

    def is_suspended(self):
        return not self._can_run.is_set()

    def resume(self):
        self._can_run.set()
        for cb in self.child_resume_cbs:
            cb()
        for child in self.child_suspendables:
            child.resume()
        warning(f"Resumed all children {self._target=}")

    async def as_coroutine(self):
        await self

class OwnerDecorator(ABC):
    _is_coroutine = asyncio.coroutines._is_coroutine

    def __init__(self, *, owned=True):
        self._owned = owned

    def __call__(self, fn):
        self._fn = fn
        if self._owned:
            return self
        else:
            return self.wrap(self._fn)

    def set_name_and_update(self, owner, name):
        if hasattr(self._fn, '__set_name__'):
            # depth-first propagation of set_name to match decorator call order
            self._fn.__set_name__(owner, name)
            # update self._fn in case other OwnerDecorators modified it
            self._fn = getattr(owner, name)

    def __set_name__(self, owner, name):
        self.set_name_and_update(owner, name)
        self.set_name(owner, name)
        wrapped_fn = self.wrap(self._fn)
        setattr(owner, name, wrapped_fn)

    @abstractmethod
    def wrap(self, fn):
        raise NotImplementedError()

    @abstractmethod
    def set_name(self, owner, name):
        raise NotImplementedError()

class async_suspendable(OwnerDecorator):
    def __init__(self, *, suspend_cb=None, resume_cb=None, **kwargs):
        super().__init__(**kwargs)
        self._suspend_cb = suspend_cb or (lambda *a, **k: None)
        self._resume_cb = resume_cb or (lambda *a, **k: None)

    def wrap(self, fn):
        @functools.wraps(fn)
        async def suspendable_coroutine(*args, **kwargs):
            return await Suspendable(fn(*args, **kwargs),
                suspend_cb=functools.partial(self._suspend_cb, *args, **kwargs),
                resume_cb=functools.partial(self._resume_cb, *args, **kwargs)
            )
        return suspendable_coroutine

    def set_name(self, owner, name):
        if isinstance(self._suspend_cb, str):
            self._suspend_cb = getattr(owner, self._suspend_cb)
        if isinstance(self._resume_cb, str):
            self._resume_cb = getattr(owner, self._resume_cb)

class sync_to_async(OwnerDecorator):
    def __init__(self, *, get_future_cb=None, done_cb=None, executor=None, **kwargs):
        super().__init__(**kwargs)
        self._get_future_cb = get_future_cb
        self._done_cb = done_cb
        self._executor = executor

    def wrap(self, fn):
        @functools.wraps(fn)
        async def run_in_executor(*args, **kwargs):
            loop = asyncio.get_running_loop()
            p_func = functools.partial(fn, *args, **kwargs)
            future = loop.run_in_executor(self._executor, p_func)
            if self._get_future_cb:
                future_cb = lambda f: self._get_future_cb(*args, **kwargs, future=f)
                future_cb(future)
            if self._done_cb:
                done_cb = lambda f: self._done_cb(*args, **kwargs, future=f)
                future.add_done_callback(done_cb)
            return await future
        return run_in_executor

    def set_name(self, owner, name):
        if isinstance(self._get_future_cb, str):
            self._get_future_cb = getattr(owner, self._get_future_cb)
        if isinstance(self._done_cb, str):
            self._done_cb = getattr(owner, self._done_cb)

def timeit(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            total_time = time.perf_counter() - start_time
            info(f'Function `{func.__name__}` took {total_time:.4f} seconds')
    return wrapper