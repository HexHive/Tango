from __future__ import annotations

from . import info, warning, critical, error
from functools import partial, wraps, cached_property
from async_property import async_property, async_cached_property
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, EnumType, auto
from collections import defaultdict
from typing import (Hashable, Optional, Mapping, Sequence, Type, TypeAlias,
    get_type_hints)
from contextvars import ContextVar, Context, Token, copy_context
import asyncio
import inspect
import time

__all__ = [
    'GLOBAL_ASYNC_EXECUTOR', 'async_property', 'cached_property',
    'async_cached_property', 'async_enumerate', 'async_suspendable',
    'sync_to_async', 'timeit', 'Suspendable', 'ComponentType', 'ComponentOwner',
    'ComponentKey', 'Configurable', 'create_session_context',
    'get_session_context', 'get_session_task_group'
]

session_context_var: ContextVar[Context] = ContextVar('session_context')
session_task_group_var: ContextVar[asyncio.TaskGroup] = ContextVar('session_task_group')

def create_session_context(tg: Optional[asyncio.TaskGroup]=None) -> Context:
    ctx = copy_context()
    ctx.run(session_context_var.set, ctx)
    ctx.run(session_task_group_var.set, tg)
    return ctx

def get_session_context() -> Context:
    try:
        return session_context_var.get()
    except LookupError as ex:
        raise RuntimeError("Not running within a session context!") from ex

def get_session_task_group() -> TaskGroup:
    try:
        return session_task_group_var.get()
    except LookupError as ex:
        raise RuntimeError("Not running within a session context!") from ex

# This is a single-threaded executor to be used for wrapping sync methods into
# async coroutines. It is single-threaded because, for ptrace to work correctly,
# the tracer must be the same thread that launched the process.
# Since ProcessLoader uses sync_to_async for _launch_target, then the parent is
# this single thread in the thread pool, and all following calls to ptrace also
# use the same thread pool (see {TCP,UDP}Channel.{send,receive}).
GLOBAL_ASYNC_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='AsyncWrapper')

class Suspendable:
    """
    Adapted from:
    https://stackoverflow.com/questions/66687549/is-it-possible-to-suspend-and-restart-tasks-in-async-python
    """
    parent: ContextVar[Suspendable] = ContextVar('suspendable_parent')

    def __init__(self, target, /, **kwargs):
        self._target = target
        self._init_kwargs = kwargs

    def _initialize(self, *, suspend_cb=None, resume_cb=None,
            sleep_task=None, wakeup_task=None):
        self._can_run = asyncio.Event()
        self._can_run.set()
        self.set_callbacks(suspend_cb=suspend_cb, resume_cb=resume_cb)
        self.set_tasks(sleep_task=sleep_task, wakeup_task=wakeup_task)
        self.child_suspendables = []

    def set_callbacks(self, **kwargs):
        def set_suspend_cb(cb):
            self._suspend_cb = cb
        def set_resume_cb(cb):
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
        self._parent_token = self.parent.set(self)
        self._parent = self._parent_token.old_value
        if self._parent is not Token.MISSING:
            self._parent.child_suspendables.append(self)

    def pop_cbs(self):
        if self._parent is not Token.MISSING:
            self._parent.child_suspendables.remove(self)
        assert self.parent.get() == self, "Parent finished without awaiting a" \
            f" Suspendable child {self.parent.get()}!"
        self.parent.reset(self._parent_token)

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
        if self._suspend_cb:
            self._suspend_cb()
        self._can_run.clear()
        warning(f"Suspended all children {self._target=}")

    def is_suspended(self):
        return not self._can_run.is_set()

    def resume(self):
        self._can_run.set()
        if self._resume_cb:
            self._resume_cb()
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
        @wraps(fn)
        async def suspendable_coroutine(*args, **kwargs):
            return await Suspendable(fn(*args, **kwargs),
                suspend_cb=partial(self._suspend_cb, *args, **kwargs),
                resume_cb=partial(self._resume_cb, *args, **kwargs)
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
        @wraps(fn)
        async def run_in_executor(*args, **kwargs):
            loop = asyncio.get_running_loop()
            p_func = partial(fn, *args, **kwargs)
            future = loop.run_in_executor(self._executor, p_func)
            if self._get_future_cb:
                self._get_future_cb(*args, **kwargs, future=future)
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

async def async_enumerate(asequence, start=0):
    """Asynchronously enumerate an async iterator from a given start value"""
    n = start
    async for elem in asequence:
        yield n, elem
        n += 1

def timeit(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            total_time = time.perf_counter() - start_time
            info(f'Function `{func.__name__}` took {total_time:.4f} seconds')
    return wrapper

###

class ComponentTypeMeta(EnumType):
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
                return cls.FakeComponentType(value)

class ComponentType(Enum, metaclass=ComponentTypeMeta):
    channel_factory = auto()
    loader = auto()
    tracker = auto()
    explorer = auto()
    input_generator = auto()
    strategy = auto()
    session = auto()

cls_hierarchy = lambda: defaultdict(cls_hierarchy)
ComponentRegistry: Mapping[ComponentKey, Mapping] = cls_hierarchy()
ComponentKey = ComponentType | str
ComponentCapture = Optional[set[ComponentKey]]
PathCapture = Optional[Sequence[str]]

# TODO make it return sets of compatible configurations instead of finding the
# first matching one
def match_component(
        component_type: ComponentKey, config: dict) -> Configurable:
    def match_component_dfs(reg) -> Optional[Configurable]:
        for base, classes in reg.items():
            match = match_component_dfs(classes)
            if match is not None:
                return match
            if base.match_config(config):
                return base
    match = match_component_dfs(ComponentRegistry[component_type])
    if match is None:
        raise LookupError(f"No valid component for `{component_type.name}`")
    return match

class ComponentOwner(ABC):
    def __init__(self, config: dict):
        self.singletons = {}
        self._config = config

    @property
    def component_classes(self):
        return {
            component_type: match_component(component_type, self._config)
            for component_type in ComponentRegistry
        }

    async def instantiate(self, component_type: ComponentKey,
            config=None, *args, **kwargs) -> TopLevelSingletonConfigurable:
        config = config or self._config
        component_type = ComponentType(component_type)
        component = await self.component_classes[component_type].instantiate( \
                        self, config, *args, **kwargs)
        return component

class TopLevelSingletonConfigurable:
    _capture_components: AnnotatedComponentCapture
    _capture_paths: PathCapture

    def __init_subclass__(cls,
            component_type: Optional[ComponentKey]=None,
            capture_components: ComponentCapture=None,
            capture_paths: PathCapture=None, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if component_type:
            component_type = ComponentType(component_type)
            if isinstance(component_type, str):
                warning(f'Using non-standard component type'
                    f' {component_type} for {cls}')

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

        mro = inspect.getmro(cls)[:0:-1]
        reg = ComponentRegistry[component_type]
        annotated_init = cls.__dict__.get('__init__')
        if capture_components and (annotated_init is None or \
                not (annot := get_type_hints(annotated_init))):
            raise TypeError(f"{cls} captures {capture_components} but does not"
                " provide an annotated __init__!")

        capture_components = capture_components or set()
        try:
            annotated_capture = {(c := (ComponentType(x)), annot[c.name])
                for x in capture_components}
        except KeyError as ex:
            raise TypeError(f"{cls}.__init__ does not provide an annotation for"
                f" captured component {ex.args[0]}!")
        cls._capture_components = annotated_capture

        if capture_paths:
            capture_paths = list(capture_paths)
        cls._capture_paths = capture_paths or list()
        for base in mro:
            if issubclass(base, TopLevelSingletonConfigurable) and \
                    base is not TopLevelSingletonConfigurable:
                if not inspect.isabstract(base):
                    reg = reg[base]
                # FIXME may be useful to keep track of where captures were
                # inherited from, especially for documentation; alternatively,
                # MRO can be resolved at instantiation instead of caching it
                cls._capture_components.update(base._capture_components)
                cls._capture_paths.extend(base._capture_paths)
        if not inspect.isabstract(cls):
            # finally, we add ourselves to the leaf of the defaultdict
            reg = reg[cls]

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return True

    async def initialize(self):
        info(f'Initializing {self}')

    @classmethod
    async def instantiate(cls, owner: ComponentOwner, config: dict,
            deps: set=None, initialize: bool=True, *args, **kwargs) \
            -> TopLevelSingletonConfigurable:
        deps = deps if deps is not None else set()
        if (typ := cls._component_type) in deps:
            raise RuntimeError(f"{typ.name} has a cyclical dependency!")

        for path in cls._capture_paths:
            for kw, value in cls.find_path(path, config):
                # WARN might overwrite values in case of globbing
                kwargs[kw] = value
        for component_type, requested_type in cls._capture_components:
            if (component := owner.singletons.get(component_type)) is None:
                component = \
                    await owner.instantiate(component_type, config, deps | {typ})
            if not component._initialized:
                error(f"{typ} attempted to capture uninitialized compononent"
                    f" {component}")
                import ipdb; ipdb.set_trace()
            if not isinstance(component, requested_type):
                raise TypeError(f"{owner} instantiated a {component_type} as an"
                    f" instance of {type(component)}, but {cls} captured an"
                    f" instance of {requested_type} instead!")
            kwargs[component_type.name] = component

        new_component = cls(*args, **kwargs)
        new_component._initialized = False
        owner.singletons[typ] = new_component
        if initialize:
            await new_component.initialize()
            new_component._initialized = True
        return new_component

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

Configurable = TopLevelSingletonConfigurable
AnnotatedComponentCapture: TypeAlias = set[tuple[
        ComponentKey, Optional[Type[Configurable]]]]