from abc import ABC
from functools import partial
from itertools import islice, tee, chain
from copy import deepcopy, copy
import inspect

class DecoratorBase(ABC):
    def __call__(self, input, copy=True): # -> InputBase:
        self._input = deepcopy(input) if copy else input
        for name, func in inspect.getmembers(self, predicate=inspect.ismethod):
            if name not in ('___iter___', '___eq___', '___add___', '___getitem___', '___repr___'):
                continue
            oldfunc = getattr(self._input, name, None)
            newfunc = partial(func, oldfunc)
            setattr(self._input, name, newfunc)

        setattr(self._input, '___decorator___', self)
        return self._input

    def ___repr___(self, orig):
        return f"{self.__class__.__name__}({orig()})"

    def undecorate(self):
        assert getattr(self, '_input') is not None, \
                "Decorator has not been called before!"
        for name, func in inspect.getmembers(self, predicate=inspect.ismethod):
            if name not in ('___iter___', '___eq___', '___add___', '___getitem___', '___repr___'):
                continue
            newfunc = getattr(self._input, name, None)
            oldfunc = newfunc.args[0]
            setattr(self._input, name, oldfunc)
        del self._input.___decorator___

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

class CachingDecorator(DecoratorBase):
    def __init__(self):
        self._cached_iter = None
        self._cached_repr = None

    def __call__(self, input, copy=True): # -> InputBase:
        self._cached_repr = repr(input)
        return super().__call__(input, copy)

    def ___iter___(self, orig):
        seq, copy = tee(orig())
        yield from seq
        self._cached_iter = tuple(copy)

        self.undecorate()
        self._input.___iter___ = self.___cached_iter___
        self._input.___repr___ = self.___cached_repr___
        del self._input

    def ___cached_iter___(self):
        yield from self._cached_iter

    def ___cached_repr___(self):
        return self._cached_repr

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
        if self.get_parent_class(input.___iter___) is self.__class__ and \
                input.___decorator___._step == self._step:
            self._input = deepcopy(input) if copy else input
            self._input.___decorator___._start += self._start
            if self._stop is not None:
                stop = self._input.___decorator___._start + (self._stop - self._start)
                assert self._input.___decorator___._stop is None or \
                    self._input.___decorator___._stop >= stop
                self._input.___decorator___._stop = stop
            return self._input
        elif self._start == 0 and self._stop == None:
            return deepcopy(input) if copy else input
        else:
            return super().__call__(input)

    def ___iter___(self, orig):
        yield from islice(orig(), self._start, self._stop, self._step)

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
        return f"{orig()}[{fmt}]"

class JoiningDecorator(DecoratorBase):
    def __init__(self, *others):
        self._others = list(others)

    def __call__(self, input, copy=True): # -> InputBase:
        if self.get_parent_class(input.___iter___) is self.__class__:
            self._input = deepcopy(input) if copy else input
            self._input.___decorator___._others.extend(self._others)
            return self._input
        else:
            return super().__call__(input)

    def ___iter___(self, orig):
        yield from chain(orig(), *self._others)

    def ___repr___(self, orig):
        first = orig()
        others = ' + '.join(repr(x) for x in self._others)
        return f'{first} + {others}'