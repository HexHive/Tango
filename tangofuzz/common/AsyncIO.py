from . import debug, warning

import asyncio
import functools
from async_property import async_property, async_cached_property
from abc import ABC, abstractmethod

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
    def __init__(self, target, /, *, suspend_cb=None, resume_cb=None):
        self._target = target
        self._can_run = asyncio.Event()
        self._can_run.set()
        self._suspend_cb = suspend_cb
        self._resume_cb = resume_cb
        self._awaited = False

        if hasattr(asyncio.current_task(), 'suspendable_ancestors'):
            self._parent = asyncio.current_task().suspendable_ancestors[-1] \
                                if asyncio.current_task().suspendable_ancestors \
                                else self
            self.child_suspendables = []
            self.child_suspend_cbs = []
            self.child_resume_cbs = []
            self._push_suspend_idx = len(self._parent.child_suspend_cbs)
            self._push_resume_idx = len(self._parent.child_resume_cbs)
            asyncio.current_task().suspendable_ancestors.append(self)
        else:
            self.push_cbs = self.pop_cbs = lambda: None

    def set_callbacks(self, *, suspend_cb=None, resume_cb=None):
        if self._awaited:
            # change the callbacks even while the coroutine gets awaited
            if self._suspend_cb:
                self._parent.child_suspend_cbs[self._push_suspend_idx] = suspend_cb
            else:
                self._parent.child_suspend_cbs.insert(self._push_suspend_idx, suspend_cb)

            if self._resume_cb:
                self._parent.child_resume_cbs[self._push_resume_idx] = resume_cb
            else:
                self._parent.child_resume_cbs.insert(self._push_resume_idx, resume_cb)
        self._suspend_cb = suspend_cb
        self._resume_cb = resume_cb

    def push_cbs(self):
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

    def __await__(self):
        self.push_cbs()

        target_iter = self._target.__await__()
        iter_send, iter_throw = target_iter.send, target_iter.throw
        send, message = iter_send, None
        # This "while" emulates yield from.
        while True:
            # wait for can_run before resuming execution of self._target
            try:
                while not self._can_run.is_set():
                    yield from self._can_run.wait().__await__()
            except BaseException as err:
                send, message = iter_throw, err

            # continue with our regular program
            try:
                signal = send(message)
            except StopIteration as err:
                self.pop_cbs()
                return err.value
            else:
                send = iter_send
            try:
                message = yield signal
            except BaseException as err:
                send, message = iter_throw, err

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
        raise NotImplemented

    @abstractmethod
    def set_name(self, owner, name):
        raise NotImplemented

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
    def __init__(self, *, get_future_cb=None, done_cb=None, **kwargs):
        super().__init__(**kwargs)
        self._get_future_cb = get_future_cb
        self._done_cb = done_cb

    def wrap(self, fn):
        @functools.wraps(fn)
        async def run_in_executor(*args, **kwargs):
            loop = asyncio.get_running_loop()
            p_func = functools.partial(fn, *args, **kwargs)
            future = loop.run_in_executor(None, p_func)
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
