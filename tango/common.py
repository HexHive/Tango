from __future__ import annotations

from . import info, warning, critical, error
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, EnumType, auto
from collections import defaultdict
from typing import (Hashable, Optional, Mapping, Sequence, Type, TypeAlias,
    Awaitable, Callable, Coroutine, AsyncIterable, AsyncIterator, Any,
    TypeVar, ParamSpec, Iterator, Iterable, get_type_hints)
from contextvars import ContextVar, Context, Token, copy_context
from concurrent.futures import Future, Executor
import numpy as np
import types
import asyncio
import inspect
import time

__all__ = [
    'GLOBAL_ASYNC_EXECUTOR', 'async_enumerate', 'async_suspendable',
    'sync_to_async', 'timeit', 'Suspendable', 'ComponentType', 'ComponentOwner',
    'ComponentKey', 'Component', 'AsyncComponent',
    'create_session_context', 'get_session_context', 'get_session_task_group'
]

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

session_context_var: ContextVar[Context] = ContextVar('session_context')
session_task_group_var: ContextVar[asyncio.TaskGroup] = ContextVar('session_task_group')

def create_session_context(tg: Optional[asyncio.TaskGroup]=None) -> Context:
    """
    Copies the current `~contextvars.Context` and sets the value of two
    `~contextvars.ContextVar` instances:
        * 'session_context': Contains a reference to the context itself. Since
          tasks created within a context get a copy of the context rather than a
          reference to it, this mechanism allows them to access variables in the
          root context. This comes in handy with
          :py:mod:`<profilers tango.core.profiler>`: a profiler may be
          instantiated within an arbitrary task, but at the root of that
          task's creation is a parent session to which all profiled values
          belong. Use :py:func:`get_session_context` to access the current
          context.
        * 'session_task_group': Contains a reference to the task group created
          by the :py:class:`~tango.fuzzer.Fuzzer`, which is passed as the
          optional parameter `tg` to this function.

    Args:
        tg (Optional[asyncio.TaskGroup], optional): If set, a
          `~asyncio.TaskGroup` object which awaits session-specific
          tasks. The caller is responsible for managing the ``TaskGroup``.

    Returns:
        Context: The session-local context.

    See Also:
        :py:func:`get_session_context`
    """
    ctx = copy_context()
    ctx.run(session_context_var.set, ctx)
    if tg:
        ctx.run(session_task_group_var.set, tg)
    return ctx

def get_session_context() -> Context:
    """
    Gets a reference to the session-local `~contextvars.Context`, which acts as
    a mapping of `~contextvars.ContextVar`s to context-specific values. These
    would serve as session-local variables, accessible within any ``Task`` in
    this session.

    Returns:
        Context: The session-local context.

    Raises:
        RuntimeError: The function was called outside a session context.

    See Also:
        :py:func:`create_session_context`
    """
    try:
        return session_context_var.get()
    except LookupError as ex:
        raise RuntimeError("Not running within a session context!") from ex

def get_session_task_group() -> TaskGroup:
    """
    Gets a reference to the session-local `~asyncio.TaskGroup`, if set within
    the current session context.

    Returns:
        TaskGroup: The `~asyncio.TaskGroup` object set at context creation.

    Raises:
        RuntimeError: The function was called outside a session context, or the
          session context was never assigned a ``TaskGroup``.
    """
    try:
        return session_task_group_var.get()
    except LookupError as ex:
        raise RuntimeError("Not running within a session context!") from ex


GLOBAL_ASYNC_EXECUTOR = ThreadPoolExecutor(thread_name_prefix='AsyncWrapper',
    max_workers=1)
"""concurrent.futures.ThreadPoolExecutor:

This is a single-threaded executor to be used for wrapping sync methods into
async coroutines, using :py:func:`sync_to_async`. It is single-threaded because,
for ``ptrace`` to work correctly, the tracer must be the same thread that
launched the process. Since :py:class:`tango.unix.ProcessLoader` uses
:py:func:`sync_to_async` to launch a process, then the parent is this single
thread in the thread pool, and all following calls to ``ptrace`` also use the
same thread pool.

See Also:
    :py:func:`tango.net.TCPChannel.send`
    :py:func:`tango.net.UDPChannel.send`
    :py:func:`tango.raw.StdIOChannel.send`
"""

class Suspendable:

    """
    An async awaitable object that wraps another awaitable object and supports
    suspending and resuming its execution. When multiple :py:class:`Suspendable`
    objects are awaited within the same `~asyncio.Task`, they form a chain of
    parent-child suspendables, where a suspension or resumption of the parent
    recurses on its children too. When a :py:class:`Suspendable` executing
    within a task creates another task, the latter inherits a copy of the
    former's `~asyncio.Context`, and any :py:class:`Suspendable` running withing
    the new task would continue the ancestry chain. The difference is that the
    parent task may create multiple tasks, thus forming an ancestry tree.

    Calling :py:func:`suspend` or `:py:func:resume:` first recurses into
    children, then calls the :py:class:`Suspendable`'s own suspension or
    resumption callbacks.

    Conversely, "sleeping" and "waking up" refer to the instants when the
    :py:class:`Suspendable` has control of the event loop, around the time its
    suspension status was modified (through :py:func:`suspend` or
    :py:func:`resume`). After suspension is requested, and as soon as the
    wrapped awaitable yields, the "sleeping" phase begins, where a sleep task,
    if specified, is awaited. Similarly, after resumption is requested, and as
    soon as the awaitable is given control of the event loop, the "waking up"
    phase begins, where the wake-up task, if specified, is awaited. This can
    come in handy in situations where suspension and resumption callbacks need
    to run asynchronously or within the context of the task (rather than the
    context of the suspender/resumer).

    Implementation was adapted from an answer to `this StackOverflow question`_.

    Args:
        aw (Awaitable): The awaitable object to be wrapped.
        suspend_cb (Optional[Callable[[], Any]], optional): The suspension
          callback to be executed in the context of the suspender.
        resume_cb (Optional[Callable[[], Any]], optional): The resumption
          callback to be executed in the context of the resumer.
        sleep_task (Optional[Awaitable], optional): The awaitable to be awaited
          during the sleeping phase, within the context of the
          :py:class:`Suspendable` itself
        wakeup_task (Optional[Awaitable], optional): The awaitable to be awaited
          during the waking up phase, within the context of the
          :py:class:`Suspendable` itself

    .. _this StackOverflow question: https://stackoverflow.com/q/66687549
    """

    parent: ContextVar[Suspendable] = ContextVar('suspendable_parent')

    def __init__(self, aw: Awaitable, /, **kwargs):
        self._aw = aw
        self._init_kwargs = kwargs

    def set_callbacks(self, *args, **kwargs):
        """
        Set or change the callbacks. To set one without modifying the other,
        only the keyword argument for the callback to be changed should be
        specified. If both callbacks need to be changed, it suffices to use
        positional arguments, in the order of (`suspend_cb`, `resume_cb`).
        If a callback needs to be cleared, a value of `None` should be passed.
        To clear both callbacks, it suffices to call :py:func:`set_callbacks`()
        without any arguments.

        Args:
            suspend_cb (Optional[Callable[[], Any]], optional): The suspension
              callback to be executed in the context of the suspender.
            resume_cb (Optional[Callable[[], Any]], optional): The resumption
              callback to be executed in the context of the resumer.
        """
        self._set_attrs(('suspend_cb', 'resume_cb'), *args, **kwargs)

    def set_tasks(self, *args, **kwargs):
        """
        Set or change the awaitable tasks. To set one without modifying the
        other, only the keyword argument for the task to be changed should be
        specified. If both tasks need to be changed, it suffices to use
        positional arguments, in the order of (`sleep_task`, `wakeup_task`). If
        a task needs to be cleared, a value of `None` should be passed. To clear
        both tasks, it suffices to call :py:func:`set_tasks`() without any
        arguments.

        Args:
            sleep_task (Optional[Awaitable], optional): The awaitable to be
              awaited during the sleeping phase, within the context of the
              :py:class:`Suspendable` itself
            wakeup_task (Optional[Awaitable], optional): The awaitable to be
              awaited during the waking up phase, within the context of the
              :py:class:`Suspendable` itself
        """
        self._set_attrs(('sleep_task', 'wakeup_task'), *args, **kwargs)

    @property
    def tasks(self) \
            -> tuple[Optional[Awaitable], Optional[Awaitable]]:
        """
        Get the current sleep and wake-up tasks, if set. `None` is returned to
        indicate unset tasks.

        Returns:
            tuple[Optional[Awaitable], Optional[Awaitable]]:
              A tuple of (`sleep_task`, `wakeup_task`).
        """
        return self._sleep_task, self._wakeup_task

    @property
    def callbacks(self) \
            -> tuple[Optional[Callable[[], Any]], Optional[Callable[[], Any]]]:
        """
        Get the current suspension and resumption callbacks, if set. `None` is
        returned to indicate unset callbacks.

        Returns:
            tuple[Optional[Callable[[], Any]], Optional[Callable[[], Any]]]:
              A tuple of (`suspend_cb`, `resume_cb`).
        """
        return self._suspend_cb, self._resume_cb

    def suspend(self):
        """
        Suspends the current awaitable target and recurses into children of the
        current :py:class:`Suspendable`, then calls its suspension callback.
        """
        for child in self.child_suspendables[::-1]:
            child.suspend()
        if self._suspend_cb:
            self._suspend_cb()
        self._can_run.clear()
        warning(f"Suspended all children {self._aw}")

    def resume(self):
        """
        Resumes the current awaitable target and recurses into children of the
        current :py:class:`Suspendable`, then calls its resumption callback.
        """
        self._can_run.set()
        if self._resume_cb:
            self._resume_cb()
        for child in self.child_suspendables:
            child.resume()
        warning(f"Resumed all children {self._aw}")

    async def as_coroutine(self) -> Coroutine:
        """
        The :py:class:`Suspendable` itself is just an awaitable object, i.e., it
        cannot be used as an argument to `asyncio.create_task`.
        :py:func:`as_coroutine` is thus a coroutine function that awaits on it,
        to enable such use-cases.

        Returns:
            Coroutine: an awaitable coroutine object.
        """
        return await self

    @property
    def suspended(self) -> bool:
        """
        bool: Indicates whether or not the awaitable target is suspended.
        """
        return not self._can_run.is_set()

    def _initialize(self, *,
            suspend_cb: Optional[Callable[[], Any]]=None,
            resume_cb: Optional[Callable[[], Any]]=None,
            sleep_task: Optional[Awaitable]=None,
            wakeup_task: Optional[Awaitable]=None):
        self._can_run = asyncio.Event()
        self._can_run.set()
        self.set_callbacks(suspend_cb=suspend_cb, resume_cb=resume_cb)
        self.set_tasks(sleep_task=sleep_task, wakeup_task=wakeup_task)
        self.child_suspendables = []

    def _set_attrs(self, names, *args, **kwargs):
        if len(args) == len(names):
            for name, value in zip(names, args):
                setattr(self, f'_{name}', value)
        elif not args and not kwargs:
            for name in names:
                setattr(self, f'{_name}', None)
        elif not args and kwargs:
            for name in names:
                try:
                    setattr(self, f'_{name}', kwargs[name])
                except KeyError:
                    pass
        else:
            raise ValueError("Attributes can only be specified entirely as"
                " positional arguments, or partially as keyword arguments."
                " Using partial positional arguments or mixing the two is not"
                " supported!")

    def _push_cbs(self):
        self._parent_token = self.parent.set(self)
        self._parent = self._parent_token.old_value
        if self._parent is not Token.MISSING:
            self._parent.child_suspendables.append(self)

    def _pop_cbs(self):
        if self._parent is not Token.MISSING:
            self._parent.child_suspendables.remove(self)
        assert self.parent.get() == self, "Parent finished without awaiting a" \
            f" Suspendable child {self.parent.get()}!"
        self.parent.reset(self._parent_token)

    def __await__(self):
        self._initialize(**self._init_kwargs)
        self._push_cbs()

        try:
            target_iter = self._aw.__await__()
            iter_send, iter_throw = target_iter.send, target_iter.throw
            send, message = iter_send, None
            # This "while" emulates yield from.
            while True:
                # wait for can_run before resuming execution of self._aw
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
            self._pop_cbs()

class OwnableDecorator(ABC):

    """
    An abstract class for function decorators that can also decorate class
    methods where, in such cases, would require access to the class in which the
    function exists, henceforth referred to as ``owner``. Optionally, the
    decorator can specify, at construction, that it is not decorating a class
    method, in which case, it behaves as a normal function wrapper.

    Such a decorator is useful where the owner's namespace is partially
    populated at decoration time (i.e. the body of the class is still being
    built) but the decorator requires access to the owner itself or the complete
    namespace.

    Stacking :py:class:`OwnableDecorator`s results in an intermediate state
    where decorated objects are not actual functions, but instances of the
    decorator itself. Once the owner is constructed, it calls ``__set_name__``
    on the decorator, which propagates a reference to the former and the name of
    the latter in its namespace into the stack, resolving the wrapped functions
    and finally updating the owner's namespace with the decorated method.

    Args:
        owned (bool, optional): Whether or not the decorator should behave as
          a class method decorator (:py:code:`owned==True`) or as a normal
          function decorator otherwise.

    See Also:
        :py:class:`async_suspendable`
        :py:class:`sync_to_async`
    """

    def __init__(self, *, owned: bool=True):
        self._owned = owned
        self._applied = False

    @abstractmethod
    def wrap(self, fn: Callable[P, Any]) -> Callable[P, Any]:
        """
        Performs the actual wrapping mechanism of the decorator.

        Args:
            fn (Callable[..., Any]): The function to be decorated.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, owner: Type, name: str):
        """
        Propagates the owner information to the decorator, giving it access to
        any attributes that would otherwise have not been accessible at
        decoration time.

        Args:
            owner (Type): A reference to the owning class.
            name (str): The name of the decorated function in the owner's
              namespace.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if not self._owned:
            # must only receive one function as an argument
            assert len(args) == 1 and not kwargs
            return self.wrap(args[0])
        elif self._applied:
            # forward the call to our wrapped function
            return self._obj(*args, **kwargs)
        else:
            if hasattr(self, '_obj'):
                raise RuntimeError("Attempted to call an un-applied decorator"
                    f" {self} of {self._obj} ({args=},{kwargs=})")
            self._obj = args[0]
            # return in intermediate state until __set_name__ is called
            return self

    def _propagate_name(self, owner: Type, name: str):
        # if the wrapped object was already an OwnableDecorator or another
        # object that implements __set_name__, we need to inform it
        if hasattr(self._obj, '__set_name__'):
            # depth-first propagation of set_name to match decorator call order
            self._obj.__set_name__(owner, name)

    def __set_name__(self, owner: Type, name: str):
        self._propagate_name(owner, name)
        self.apply(owner, name)
        self._applied = True
        self._obj = self.wrap(self._obj)

    def __get__(self, obj, owner):
        if obj:
            return types.MethodType(self._obj, obj)
        else:
            return self

class LazyReferenceDecorator(OwnableDecorator):

    """
    An :py:class:`OwnableDecorator` that accepts `str` references to owner
    attributes at decoration time, then resolves them to actual object
    references at apply time, i.e. when the owner's namespace is
    complete.

    Subclasses must implement the :py:func:`attributes` property that returns a
    list or iterable of attribute names to capture from the constructor at
    decoration time, then resolve from owner and assign to `self` at apply time.
    """

    def __init__(self, **kwargs):
        for attr in self.attributes:
            try:
                setattr(self, attr, kwargs.pop(attr))
            except KeyError:
                # ask for forgiveness, not permission
                pass
        super().__init__(**kwargs)

    @classmethod
    @property
    @abstractmethod
    def attributes(cls) -> Iterable[str]:
        pass

    def _apply_attr(self, attr, owner):
        if isinstance(v := getattr(self, attr), str):
            setattr(self, attr, getattr(owner, v))

    def apply(self, owner, name):
        for attr in self.attributes:
            self._apply_attr(attr, owner)

class async_suspendable(LazyReferenceDecorator):

    """
    An :py:class:`OwnableDecorator` that wraps the underlying coroutine
    function in a :py:class:`Suspendable`. The specified callbacks and tasks
    receive the calling arguments of the wrapped function (including a
    reference to an instance's ``self`` variable as the first positional
    argument); they must have a call signature that is a superset of that of
    the wrapped function. Addionally, tasks must be callable coroutine
    functions; it is no longer enough for the tasks to be awaitable, they
    must instead be callables that return an awaitable.

    Callbacks and tasks can directly refer to functions within the class
    body of the owner, or can be specified as the name of the object to be
    resolved from the owner when its namespace is complete.

    Args:
        suspend_cb (Optional[Callable[..., Any] | str], optional):
          The suspension callback.
        resume_cb (Optional[Callable[..., Any] | str], optional):
          The resumption callback.
        sleep_task (Optional[Callable[..., Coroutine] | str], optional):
          The sleep task coroutine function.
        wakeup_task (Optional[Callable[..., Coroutine] | str], optional):
          The wake-up task coroutine function.
        **kwargs: Passed to :py:func:`LazyReferenceDecorator.__init__`
    """

    _attrs = ('suspend_cb', 'resume_cb', 'sleep_task', 'wakeup_task')

    def __init__(self, *,
            suspend_cb: Optional[Callable[..., Any] | str]=None,
            resume_cb: Optional[Callable[..., Any] | str]=None,
            sleep_task: Optional[Callable[..., Coroutine] | str]=None,
            wakeup_task: Optional[Callable[..., Coroutine] | str]=None,
            **kwargs):
        # this is kept for potential type-checking in the future
        super().__init__(suspend_cb=suspend_cb, resume_cb=resume_cb,
            sleep_task=sleep_task, wakeup_task=wakeup_task, **kwargs)

    @classmethod
    @property
    def attributes(cls) -> tuple[str]:
        return cls._attrs

    def wrap(self, fn):
        @wraps(fn)
        async def suspendable_coroutine(*args, **kwargs):
            return await Suspendable(fn(*args, **kwargs),
                **self._construct_partials(*args, **kwargs)
            )
        return suspendable_coroutine

    def _construct_partials(self, *args, **kwargs):
        kw = {}
        for attr in self.attributes:
            if (v := getattr(self, attr, None)):
                kw[attr] = partial(v, *args, **kwargs)
        return kw

class sync_to_async(LazyReferenceDecorator):

    """
    An :py:class:`OwnableDecorator` that wraps the underlying synchronous
    function in a call to `asyncio.run_in_executor`, with the added feature of
    calling `set_future_cb` with the `~asyncio.Future` as argument, to allow
    other parties to await or inspect it. It also calls `done_cb` when the
    function returns. The callbacks receive the calling arguments of the wrapped
    function (including a reference to an instance's ``self`` variable as the
    first positional argument); they must have a call signature that is a
    superset of that of the wrapped function

    Args:
        executor (Optional[Executor], optional):
          The `~concurrent.futures.Executor` to use for scheduling the function.
          If unspecified, the default `asyncio` loop executor will be used.
        set_future_cb (Optional[Callable[[..., Future], None] | str], optional):
          The callback to receive the `Future` object as returned from
          `loop.run_in_executor`, before it is awaited.
        done_cb (Optional[Callable[[..., Future], None] | str], optional):
          The callback which is called once the result of the `Future` is ready,
          i.e. once the wrapped function has finished execution.
        **kwargs: Passed to :py:func:`LazyReferenceDecorator.__init__`
    """

    _attrs = ('set_future_cb', 'done_cb')

    def __init__(self, *,
            executor: Optional[Executor]=None,
            set_future_cb: Optional[Callable[[..., Future], None] | str]=None,
            done_cb: Optional[Callable[[..., Future], None] | str]=None,
            **kwargs):
        self._executor = executor
        # this is kept for potential type-checking in the future
        super().__init__(set_future_cb=set_future_cb, done_cb=done_cb, **kwargs)

    @classmethod
    @property
    def attributes(cls) -> tuple[str]:
        return cls._attrs

    def wrap(self, fn):
        @wraps(fn)
        async def run_in_executor(*args, **kwargs):
            loop = asyncio.get_running_loop()
            p_func = partial(fn, *args, **kwargs)
            future = loop.run_in_executor(self._executor, p_func)
            if self.set_future_cb:
                self.set_future_cb(*args, **kwargs, future=future)
            if self.done_cb:
                done_cb = lambda f: self.done_cb(*args, **kwargs, future=f)
                future.add_done_callback(done_cb)
            return await future
        return run_in_executor

async def async_enumerate(asequence: AsyncIterable[T], start: int=0) \
        -> AsyncIterator[tuple[int, T]]:
    """
    Asynchronously enumerate an async iterable from a given start value

    Args:
        asequence (AsyncIterable[T]): The async iterable to enumerate.
        start (int, optional): The index corresponding to the first element.

    Yields:
        AsyncIterator[tuple[int, T]]: An async-equivalent enumerate() iterator.
    """
    n = start
    async for elem in asequence:
        yield n, elem
        n += 1

def timeit(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """
    A simple performance timing wrapper for coroutine functions. The wrapped
    coroutine function measures the time it takes for the underlying coroutine
    to finish execution, then prints out the time it took on the `root` logger,
    with a logging.INFO level.

    Args:
        func (Callable[P, Awaitable[R]]): The coroutine function to wrap.

    Returns:
        Callable[P, Awaitable[R]]: The wrapped coroutine function.
    """
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            total_time = time.perf_counter() - start_time
            info(f'Function `{func.__name__}` took {total_time:.4f} seconds')
    return wrapper

###

class ComponentTypeMeta(EnumType):
    _fake_component_cache = {}
    class FakeComponentType:
        def __init__(self, value):
            self._value = value
        def __str__(self):
            return str(self._value)
        def __hash__(self):
            return hash(self._value)
        def __repr__(self):
            return f'{self.__class__}({str(self)})'
        def __eq__(self, other: Any):
            return hasattr(other, 'name') and self.name == other.name
        @property
        def name(self):
            return str(self)
    def __call__(cls, value, *args, **kwargs):
        try:
            return super().__call__(value, *args, **kwargs)
        except ValueError:
            try:
                return super().__getitem__(value)
            except KeyError:
                default = cls.FakeComponentType(value)
                cached = cls._fake_component_cache.setdefault(value, default)
                if cached is default:
                    warning(f"Using non-standard component type `{value}`")
                return cached

class ComponentType(Enum, metaclass=ComponentTypeMeta):

    """
    An Enum of core configurable components, used for component discovery and
    instantiation. It also supports resolving string literals to instances of
    :py:class:`ComponentType` by calling `ComponentType(str_lit)`. String
    literals should match the `name` attribute of instances of
    py:class:`ComponentType`. Otherwise, a string-based `ComponentType`-like
    object is returned, and a warning is logged on the `root` logger.

    Attributes:
        channel_factory
        driver
        tracker
        loader
        explorer
        generator
        strategy
        session

    See Also:
        :py:class:`ComponentOwner`
        :py:class:`Component`
        :py:class:`AsyncComponent`
    """

    channel_factory = auto()
    driver = auto()
    tracker = auto()
    loader = auto()
    explorer = auto()
    generator = auto()
    strategy = auto()
    session = auto()

class Component:
    """
    Todo:
        * Implement a non-async version of configurable components
    """
    def __init__(self, **kwargs):
        if kwargs:
            warning(f'{self} received but did not capture {kwargs=}')

    @classmethod
    def match_config(cls, config: dict) -> bool:
        """
        Inspects the configuration dict and returns whether or not the requested
        parameters match the current component. If matched, the component class
        is added as a possible candidate for instantiation of its component
        type.

        Args:
            config (dict): The configuration dict.

        Returns:
            bool: Whether or not the configuration matches the component.
        """
        return True

ComponentKey = ComponentType | str
"""TypeAlias:

This type alias represents the data type that can be used for specifying
component types, for registration or for capture.
"""

ComponentCapture = Optional[set[ComponentKey]]
"""TypeAlias:

This type alias represents the data type that component capture parameters must
have in the :py:func:`AsyncComponent.__init_subclass__` function.
"""

PathCapture = Optional[Iterable[str]]
"""TypeAlias:

This type alias represents the data type that path capture parameters must have
in the :py:func:`AsyncComponent.__init_subclass__` function.
"""

recursive_dict = lambda: defaultdict(recursive_dict)
ComponentRegistry: Mapping[ComponentKey, Mapping] = recursive_dict()
ComponentCatchAll: Mapping[ComponentKey, Component] = dict()

class ComponentOwner(dict, ABC):

    """
    A component owner is an object that holds singleton instances of main fuzzer
    components for use throughout the lifetime of a session. It provides the
    configuration dict, which components can inspect to match against the chosen
    parameters. Subclasses of :py:class:`ComponentOwner` can provide the config
    dict through other means or override :py:func:`match_component` to implement
    more complex matching criteria (e.g. compatible type annotations).

    For successful component discovery, the configuration dict should contain
    keys and values that match and satisfy those defined by
    :py:func:`Component.match_config`. Furthermore, components can specify
    configuration parameters through captured paths within the dict. A path is a
    dot-separated sequence of nested dict keys (e.g. 'channel.tcp.port' ===
    config['channel']['tcp']['port']). A configuration dict must thus provide
    these values for proper instantiation of the matched component, unless the
    component defines defaults otherwise.

    Args:
        config (dict): A dict (or mapping) of keys to values or other
            sub-dicts of configuration.

    See Also:
        :py:class:`tango.fuzzer.FuzzerConfig`
    """

    def __init__(self, config: dict):
        self._config = recursive_dict()
        self._config.update(config)
        self._initialized = set()
        self._finalized = set()

    __repr__ = object.__repr__

    def __getitem__(self, key: ComponentKey):
        key = ComponentType(key)
        return super().__getitem__(key)

    @property
    def component_classes(self) -> Mapping[ComponentKey, Component]:
        """dict:
        A lazy-evaluated dict mapping :py:obj:`ComponentKey` to
        :py:class:`Component` classes, to be used as instance factories.
        Lazy evaluation is used since some components may be registered after
        the :py:class:`ComponentOwner` instance had been created (e.g. late
        import), in which case a greedy evaluation would miss the newly-added
        components.
        """
        class hashabledict(dict):
            def __key(self):
                return tuple(self.items())
            def __hash__(self):
                return hash(self.__key())
            def __eq__(self, other):
                return self.__key() == other.__key()
        matches = {
            component_type: tuple(
                self.match_component(component_type, self._config))
            for component_type in ComponentRegistry
        }
        kls_map = self.resolve_dependencies(hashabledict(matches))
        return kls_map

    @classmethod
    @cache
    def resolve_dependencies(cls,
            matches: Mapping[ComponentKey, Sequence[Component]]) \
            -> dict[ComponentKey, Component]:
        def verify_combination(*idx):
            valid = np.zeros(len(types), dtype=bool)
            for i, m in enumerate(idx):
                typ = types[i]
                kls = matches[typ][m]
                for t, kls_c in kls._capture_components.items():
                    try:
                        j = types.index(t)
                        kls_m = matches[t][idx[j]]
                    except ValueError:
                        break
                    if not issubclass(kls_m, kls_c):
                        break
                else:
                    valid[i] = True
            return np.all(valid)
        types = list(matches.keys())
        u_verify_combination = np.frompyfunc(verify_combination, len(types), 1)
        shape = [len(matches[t]) for t in types]
        grid = np.indices(shape)
        valid = zip(*u_verify_combination(*grid).nonzero())
        combinations = []
        for combination in valid:
            combinations.append({
                (t:=types[i]): matches[t][c] for i, c in enumerate(combination)
            })
        if not combinations:
            raise TypeError("No valid combination of components found!")
        combination = combinations.pop(0)
        if combinations:
            warning("More than one valid configuration possible." \
                f" Choosing: {list(combination.values())}")
        return combination

    async def instantiate(self, component_type: ComponentKey, *args,
            config: Optional[dict]=None, **kwargs) -> AsyncComponent:
        """
        Calls the :py:func:`Component.instantiate` couroutine, supplying `self`
        as the owner, and `self._config` as the configuration dict if left
        unspecified. Passing a reference to `self` allows components to query
        the owner for dependencies. For example, if `strategy` and `session`
        both depend on `generator`, then `generator` must be instantiated while
        one of its dependents is being instantiated. Meanwhile, when the other
        dependent queries the owner, it finds an existing instance of
        `generator`, ready to be captured.

        Args:
            component_type (ComponentKey): The component type to instantiate.
            config (Optional[dict], optional): The configuration dict.
            *args: Positional arguments to pass to the component constructor.
            **kwargs: Keyword argumentsto pass to the component constructor,
              in addition to those from the captured paths.

        Returns:
            AsyncComponent: A singleton instance of the requested component.
        """
        config = config or self._config
        component_type = ComponentType(component_type)
        component = await self.component_classes[component_type].instantiate(
                        self, config, *args, **kwargs)
        return component

    @staticmethod
    def match_component(
            component_type: ComponentKey, config: dict) -> Sequence[Component]:
        """
        Todo:
            * Make it return sets of compatible configurations instead of
              finding the first matching one
        """
        def match_component_dfs(reg, catch_all: list) -> Iterator[Component]:
            for base, classes in reg.items():
                yield from match_component_dfs(classes, catch_all)
                if inspect.isabstract(base):
                    continue
                if not base._catch_all and base.match_config(config):
                    yield base
                elif base._catch_all:
                    catch_all.append(base)
        catch_all = []
        matches = list(match_component_dfs(ComponentRegistry[component_type],
            catch_all)) + catch_all[::-1]
        if not matches:
            raise LookupError(f"No valid component for `{component_type.name}`")
        return matches

class AsyncComponent(Component):
    _capture_components: AnnotatedComponentCapture
    _capture_paths: PathCapture
    _catch_all: bool = False

    def __init_subclass__(cls,
            component_type: Optional[ComponentKey]=None,
            capture_components: ComponentCapture=None,
            capture_paths: PathCapture=None,
            catch_all: bool=False, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if component_type:
            component_type = ComponentType(component_type)

        if not catch_all:
            cls._catch_all = False
        elif inspect.isabstract(cls):
            raise TypeError(f"Abstract class {cls} registered as a catch_all"
                f" for {component_type}!")
        elif (prev := ComponentCatchAll.setdefault(component_type, cls)) != cls:
            raise TypeError(f"{cls} registered as a catch_all for"
                f" {component_type}, for which {prev} is already a catch_all.")
        else:
            cls._catch_all = True

        base_type = getattr(cls, '_component_type', None)
        component_type = component_type or base_type
        if base_type and base_type != component_type:
            raise TypeError(f"{cls} registered as a {component_type} when it"
                f" inherits from a {base_type}!")
        elif not component_type:
            raise TypeError(f"Cannot create configurable {cls} without a"
                " component_type in any of its bases.")
        elif not base_type:
            cls._component_type = component_type

        # inherit capture specs from bases
        mro = inspect.getmro(cls)[:0:-1]
        reg = ComponentRegistry[component_type]
        inherit_components = dict()
        inherit_paths = set()
        for base in mro:
            if issubclass(base, AsyncComponent) and \
                    base is not AsyncComponent:
                reg = reg[base]
                # FIXME may be useful to keep track of where captures were
                # inherited from, especially for documentation; alternatively,
                # MRO can be resolved at instantiation instead of caching it
                inherit_components.update(base._capture_components)
                inherit_paths.update(base._capture_paths)

        annotated_init = cls.__dict__.get('__init__')
        if capture_components and (annotated_init is None or \
                not (annot := get_type_hints(annotated_init))):
            raise TypeError(f"{cls} captures {capture_components} but does not"
                " provide an annotated __init__!")

        capture_components = capture_components or set()
        try:
            annotated_capture = {(c := ComponentType(x)): annot[c.name]
                for x in capture_components}
        except KeyError as ex:
            raise TypeError(f"{cls}.__init__ does not provide an annotation for"
                f" captured component {ex.args[0]}!")
        capture_components = annotated_capture or \
            cls.__dict__.get('_capture_components', dict())

        if capture_paths:
            capture_paths = set(capture_paths)
        capture_paths = capture_paths or \
            cls.__dict__.get('_capture_paths', set())

        # override capture specs with current class
        cls._capture_components = inherit_components | capture_components
        cls._capture_paths = inherit_paths | capture_paths

        if not inspect.isabstract(cls):
            # finally, we add ourselves to the leaf of the defaultdict
            reg = reg[cls]

    async def initialize(self):
        info(f'Initializing {self}')

    async def finalize(self, owner: ComponentOwner):
        info(f'Finalized {self}')

    @classmethod
    async def instantiate(cls, owner: ComponentOwner, config: dict, *args,
            dependants: set=None, initialize: bool=True, finalize: bool=True,
            **kwargs) -> AsyncComponent:
        if finalize and not initialize:
            raise ValueError("It is invalid to finalize without initializing!")
        deps = dependants or set()
        if (typ := cls._component_type) in deps:
            raise RuntimeError(f"{typ.name} has a cyclical dependency!")

        if (new_component := owner.get(cls._component_type)):
            assert new_component in owner._initialized and \
                new_component in owner._finalized
            return new_component

        for path in cls._capture_paths:
            for kw, value in cls.find_path(path, config):
                # WARN might overwrite values in case of globbing
                kwargs[kw] = value
        for component_type, requested_type in cls._capture_components.items():
            if (component := owner.get(component_type)) is None:
                component = \
                    await owner.instantiate(component_type, config=config,
                        dependants=deps | {typ})
            if not isinstance(component, requested_type):
                raise TypeError(f"{owner} instantiated a {component_type} as an"
                    f" instance of {type(component)}, but {cls} captured an"
                    f" instance of {requested_type} instead!")
            kwargs[component_type.name] = component

        strargs = (str(a) for a in args)
        strkwargs = ('='.join(str(x) for x in item) for item in kwargs.items())
        _args = ', '.join((*strargs, *strkwargs))
        info(f'Creating {typ.name} as {cls.__name__}({_args})')
        new_component = cls(*args, **kwargs)
        owner[typ] = new_component
        if not deps:
            if initialize:
                # initialization is breadth-first
                await new_component.initialize_dependencies(owner)
            if finalize:
                # finalization is depth-first
                await new_component.finalize_dependencies(owner)
        return new_component

    @classmethod
    async def initialize_dependencies(cls, owner: ComponentOwner,
            preinitialize: Optional[Callable[[AsyncComponent]], None]=None):
        queue = [cls]
        while queue:
            kls = queue.pop(0)
            component = owner[kls._component_type]
            if not component in owner._initialized:
                if preinitialize:
                    preinitialize(component)
                await component.initialize()
                owner._initialized.add(component)
            queue.extend(component._capture_components.values())

    @classmethod
    async def finalize_dependencies(cls, owner: ComponentOwner,
            postfinalize: Optional[Callable[[AsyncComponent]], None]=None):
        stack = [cls]
        visited = set()
        while stack:
            kls = stack.pop()
            component = owner[kls._component_type]
            if kls in visited:
                if not component in owner._finalized:
                    await component.finalize(owner)
                    owner._finalized.add(component)
                    if postfinalize:
                        postfinalize(component)
            else:
                stack.append(kls)
                stack.extend(component._capture_components.values())
                visited.add(kls)

    @staticmethod
    def find_path(path: str, config: dict, *, expand_globs: bool=True):
        def find_path_rec(parts: list, subdict: dict, *, expand_glob=True):
            part = parts[0]
            if part == '*' and expand_glob:
                for key, value in subdict.items():
                    parts = [key] + parts[1:]
                    yield from find_path_rec(parts, subdict, expand_glob=False)
            else:
                if not part in subdict:
                    return
                elif len(parts) == 1:
                    yield part, subdict[part]
                else:
                    yield from find_path_rec(parts[1:], subdict[part],
                        expand_glob=expand_globs)

        parts = path.split('.')
        yield from find_path_rec(parts, config, expand_glob=expand_globs)

AnnotatedComponentCapture = Mapping[ComponentKey,
    Optional[Type[AsyncComponent]]]