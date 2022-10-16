from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union
from functools import partial
import inspect
from itertools import islice, chain
from profiler import ProfileValueMean

class InputBase(ABC):
    _COUNTER = 0

    def __init__(self):
        self.id = self.uniq_id
        self._decoration_depth = 0

    @abstractmethod
    def ___iter___(self):
        pass

    def ___repr___(self):
        return f"{self.__class__.__name__}:0x{self.id:08X}"

    def ___len___(self):
        return None

    async def ___aiter___(self):
        for e in iter(self):
            yield e

    def ___eq___(self, other):
        diff = False

        def zip_strict(*iterators):
            for x in zip(*iterators):
                yield x
            failed = False
            for x in iterators:
                try:
                    next(x)
                    failed = True
                    break
                except Exception:
                    pass
            return failed

        def eq(*iterators):
            nonlocal diff
            diff = yield from zip_strict(*iterators)

        return all(type(x) == type(y) and x == y
                    for x, y in eq(iter(self), iter(other))) and \
               not diff

    def ___getitem___(self, idx: Union[int, slice]):
        return SlicingDecorator(idx)(self)

    def __add__(self, other: InputBase):
        return JoiningDecorator(other)(self)

    def __iadd__(self, other: InputBase):
        return JoiningDecorator(other)(self, copy=False)

    @classmethod
    @property
    def uniq_id(cls):
        cls._COUNTER += 1
        return cls._COUNTER

    ## Decoratable functions ##
    # These definitions are needed so that a decorator can override the behavior
    # of "special methods". The python interpreter does not resolve special
    # methods as it would other class attributes, so a level of indirection is
    # needed to bypass this limitation.
    def __iter__(self):
        return self.___iter___()

    def __repr__(self):
        return self.___repr___()

    def __len__(self):
        return self.___len___()

    def __aiter__(self):
        return self.___aiter___()

    def __eq__(self, other):
        return self.___eq___(other)

    def __getitem__(self, idx: Union[int, slice]):
        return self.___getitem___(idx)

class DecoratorBase(ABC):
    DECORATED_METHODS = ('___iter___', '___aiter___', '___eq___', '___add___',
                         '___getitem___', '___repr___', '___len___')
    DECORATOR_MAX_DEPTH = 10

    def __call__(self, input, copy=True) -> InputBase:
        self._handle_copy(input, copy)

        for name, func in inspect.getmembers(self, predicate=inspect.ismethod):
            if name not in self.DECORATED_METHODS:
                continue
            oldfunc = getattr(input, name, None)
            newfunc = partial(func, oldfunc)
            setattr(self._input, name, newfunc)

        self._input._decoration_depth = input._decoration_depth + 1
        ProfileValueMean("decorator_depth", samples=100)(self._input._decoration_depth)

        if self._input._decoration_depth > self.DECORATOR_MAX_DEPTH:
            return MemoryCachingDecorator()(self._input, copy=copy)
        else:
            return self._input

    def _handle_copy(self, input, copy):
        self._input_id = input.id
        self._input = DecoratedInput(self)

    def ___repr___(self, orig):
        return f"{self.__class__.__name__}({orig()})"

    def undecorate(self):
        assert getattr(self, '_input', None) is not None, \
                "Decorator has not been called before!"
        for name, func in inspect.getmembers(self, predicate=inspect.ismethod):
            if name not in self.DECORATED_METHODS:
                continue
            newfunc = getattr(self._input, name, None)
            # extract `orig` from the partial function
            oldfunc = newfunc.args[0]
            delattr(self._input, name)
            setattr(self._input, name, oldfunc)
        del self._input.___decorator___
        del self._input

    @classmethod
    def get_parent_class(cls, method):
        """
        Gets the class that defined the method.
        Courtesy of:
        https://stackoverflow.com/questions/3589311/get-defining-class-of-unbound-method-object-in-python-3/25959545#25959545
        """
        if isinstance(method, partial):
            return cls.get_parent_class(method.func)
        if inspect.ismethod(method) or (inspect.isbuiltin(method) and getattr(method, '__self__', None) is not None and getattr(method.__self__, '__class__', None)):
            for cls in inspect.getmro(method.__self__.__class__):
                if method.__name__ in cls.__dict__:
                    return cls
            method = getattr(method, '__func__', method)  # fallback to __qualname__ parsing
        return getattr(method, '__objclass__', None)  # handle special descriptor objects

class SlicingDecorator(DecoratorBase):
    def __init__(self, idx):
        self._idx = idx
        if isinstance(self._idx, slice):
            self._start = self._idx.start or 0
            self._stop = self._idx.stop
            self._step = self._idx.step or 1
        else:
            self._start = self._idx
            self._stop = self._idx + 1
            self._step = 1

    def __call__(self, input, copy=True): # -> InputBase:
        if self._start == 0 and self._stop == None:
            return input
        elif self.get_parent_class(input.___iter___) is self.__class__ and \
                input.___decorator___._step == self._step:
            self._start += input.___decorator___._start
            if self._stop is not None:
                self._stop = input.___decorator___._start + (self._stop - self._start)
                assert input.___decorator___._stop is None or \
                    input.___decorator___._stop >= self._stop
            else:
                self._stop = input.___decorator___._stop

        return super().__call__(input)

    def ___iter___(self, orig):
        return islice(orig(), self._start, self._stop, self._step)

    def ___repr___(self, orig):
        if self._stop == self._start + 1 and self._step == 1:
            fmt = f'{self._start}'
        elif self._stop is None:
            if self._step == 1:
                fmt = f'{self._start}:'
            else:
                fmt = f'{self._start}::{self._step}'
        else:
            fmt = f'{self._start}:{self._stop}:{self._step}'
        return f'SlicedInput:0x{self._input.id:08X} (0x{self._input_id:08X}[{fmt}])'

class JoiningDecorator(DecoratorBase):
    def __init__(self, *others):
        self._others = list(others)

    def __call__(self, input, copy=True): # -> InputBase:
        if self.get_parent_class(input.___iter___) is self.__class__:
            self._others = input.___decorator___._others + self._others
        return super().__call__(input)

    def ___iter___(self, orig):
        return chain(orig(), *self._others)

    def ___repr___(self, orig):
        id = f'0x{self._input_id:08X}'
        ids = (f'0x{x.id:08X}' for x in self._others)
        return f'JoinedInput:0x{self._input.id:08X} ({" || ".join((id, *ids))})'

class MemoryCachingDecorator(DecoratorBase):
    def __init__(self):
        self._cached_iter = None
        self._cached_repr = None

    def __call__(self, input, copy=True): # -> InputBase:
        self._handle_copy(input, copy)

        self._cached_repr = repr(input)
        self._cached_iter = tuple(input)
        self._cached_len = len(self._cached_iter)
        self._input.___repr___ = self.___cached_repr___
        self._input.___iter___ = self.___cached_iter___
        self._input.___len___ = self.___cached_len___
        self._input._decoration_depth = 1

        return self._input

    def undecorate(self):
        assert getattr(self, '_input', None) is not None, \
                "Decorator has not been called before!"
        del self._input.___decorator___
        del self._input

    def ___cached_iter___(self):
        return iter(self._cached_iter)

    def ___cached_repr___(self):
        return self._cached_repr

    def ___cached_len___(self):
        return self._cached_len

class DecoratedInput(InputBase):
    def __init__(self, decorator: DecoratorBase):
        super().__init__()
        self.___decorator___ = decorator

    def ___iter___(self):
        raise NotImplementedError()

    def __del__(self):
        self.___decorator___.undecorate()
