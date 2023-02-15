from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
from typing import Union
from functools import partial, partialmethod
import inspect
from itertools import islice, chain
from profiler import ProfileValueMean
import types

class InputBase(ABC):
    _COUNTER = 0

    def __init__(self):
        self.id = self.uniq_id

    @abstractmethod
    def ___iter___(self):
        pass

    def ___repr___(self):
        return f"{self.__class__.__name__}:0x{self.id:08X}"

    def ___len___(self):
        # if NotImplementedError is raised, then tuple(input) fails;
        # if None is returned, tuple resorts to dynamic allocation
        return None

    def __bool__(self):
        # we define this so that __len__ is not used to test truthiness
        return True

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

    def ___add___(self, other: InputBase):
        return JoiningDecorator(other)(self)

    @classmethod
    @property
    def uniq_id(cls):
        cls._COUNTER += 1
        return cls._COUNTER

    @property
    def decorated(self):
        return False

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

    def __add__(self, other: InputBase):
        return self.___add___(other)

class DecoratorBaseMeta(ABCMeta):
    ALL_DECORATABLE_METHODS = {'___iter___', '___aiter___', '___len___',
        '___add___', '___getitem___', '___repr___', '___eq___'}

    def __new__(metacls, name, bases, namespace):
        if 'DECORATABLE_METHODS' not in namespace:
            kls_dec_methods = metacls.ALL_DECORATABLE_METHODS & namespace.keys()
            for base in bases:
                if hasattr(base, 'DECORATABLE_METHODS'):
                    kls_dec_methods |= base.DECORATABLE_METHODS
            namespace['DECORATABLE_METHODS'] = kls_dec_methods
        kls = super().__new__(metacls, name, bases, namespace)
        return kls

class DecoratorBase(ABC, metaclass=DecoratorBaseMeta):
    DECORATOR_MAX_DEPTH = 10

    def __call__(self, input, copy=True) -> InputBase:
        self._handle_copy(input, copy)

        for name in self.DECORATABLE_METHODS:
            func = getattr(self, name)
            oldfunc = getattr(input, name, None)
            newfunc = partialmethod(partial(func), oldfunc).__get__(self._input)
            setattr(self._input, name, newfunc)

        if self._input.___decorator_depth___ > self.DECORATOR_MAX_DEPTH:
            return MemoryCachingDecorator()(self._input, copy=copy)
        else:
            return self._input

    def _handle_copy(self, input, copy) -> InputBase:
        self._input_id = input.id
        if input.decorated:
            depth = input.___decorator_depth___ + 1
        else:
            depth = 1
        self._input = DecoratedInput(self, depth)
        ProfileValueMean("decorator_depth", samples=100)(depth)
        return self._input

    def ___repr___(self, input, orig):
        return f"{self.__class__.__name__}({orig()})"

    def undecorate(self):
        assert getattr(self, '_input', None) is not None, \
                "Decorator has not been called before!"
        for name in self.DECORATABLE_METHODS:
            newfunc = getattr(self._input, name, None)
            # extract `orig` from the partial function
            oldfunc = newfunc._partialmethod.args[0]
            setattr(self._input, name, oldfunc)
        self._input.___decorator___ = None
        self._input = None

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

    def pop(self) -> InputBase:
        if not self.DECORATABLE_METHODS or not hasattr(self, '_input'):
            # an "empty" decorator decorates no inputs
            return None
        # a decorator usually decorates at least one method, which allows us to
        # extract the `orig` argument and return its owner
        decorated_method = getattr(self._input, next(iter(self.DECORATABLE_METHODS)))
        # decorated methods are partial functions with args[0] == orig
        orig = decorated_method._partialmethod.args[0]
        owner = orig.__self__
        return owner

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
        elif input.decorated and isinstance(input.___decorator___, self.__class__) \
                and input.___decorator___._step == self._step:
            input, other = input.pop_decorator()
            self._start += other._start
            if self._stop is not None:
                self._stop = other._start + (self._stop - self._start)
                assert other._stop is None or \
                    other._stop >= self._stop
            else:
                self._stop = other._stop

        return super().__call__(input)

    def ___iter___(self, input, orig):
        return islice(orig(), self._start, self._stop, self._step)

    def ___repr___(self, input, orig):
        if self._stop == self._start + 1 and self._step == 1:
            fmt = f'{self._start}'
        elif self._stop is None:
            if self._step == 1:
                fmt = f'{self._start}:'
            else:
                fmt = f'{self._start}::{self._step}'
        else:
            fmt = f'{self._start}:{self._stop}:{self._step}'
        return f'SlicedInput:0x{input.id:08X} (0x{self._input_id:08X}[{fmt}])'

class JoiningDecorator(DecoratorBase):
    def __init__(self, *others):
        self._others = list(others)

    def __call__(self, input, copy=True): # -> InputBase:
        if input.decorated and isinstance(input.___decorator___, self.__class__):
            input, other = input.pop_decorator()
            self._others = other._others + self._others
        return super().__call__(input)

    def ___iter___(self, input, orig):
        return chain(orig(), *self._others)

    def ___repr___(self, input, orig):
        id = f'0x{self._input_id:08X}'
        ids = (f'0x{x.id:08X}' for x in self._others)
        return f'JoinedInput:0x{input.id:08X} ({" || ".join((id, *ids))})'

class MemoryCachingDecorator(DecoratorBase):
    def __call__(self, input, copy=True): # -> InputBase:
        new_input = self._handle_copy(input, copy)

        new_input._cached_repr = repr(input)
        new_input._cached_iter = tuple(input)
        new_input._cached_len = len(new_input._cached_iter)

        cls = self.__class__
        for name in self.DECORATABLE_METHODS:
            func = getattr(cls, name)
            new_func = types.MethodType(func, new_input)
            setattr(new_input, name, new_func)

        self.undecorate()
        return new_input

    def undecorate(self):
        assert getattr(self, '_input', None) is not None, \
                "Decorator has not been called before!"
        self._input.___decorator_depth___ = 0
        self._input.___decorator___ = None
        self._input = None

    def pop(self) -> InputBase:
        raise NotImplementedError("This should never be reachable!")

    def ___iter___(self):
        return iter(self._cached_iter)

    def ___repr___(self):
        return self._cached_repr

    def ___len___(self):
        return self._cached_len

class DecoratedInput(InputBase):
    def __init__(self, decorator: DecoratorBase, depth: int=0):
        super().__init__()
        self.___decorator___ = decorator
        self.___decorator_depth___ = depth

    def ___iter___(self):
        raise NotImplementedError()

    def __del__(self):
        if self.decorated:
            self.___decorator___.undecorate()

    @property
    def decorated(self):
        return (self.___decorator_depth___ > 0) or self.___decorator___

    def pop_decorator(self) -> (InputBase, DecoratorBase):
        decorator = self.___decorator___
        return decorator.pop(), decorator

    def search_decorator_stack(self, select: Callable[[DecoratorBase], bool], max_depth: int=None) -> DecoratorBase:
        depth = 0
        input = self
        while True:
            if not input.decorated:
                raise RuntimeError("Reached end of decorator stack")
            elif max_depth and depth >= max_depth:
                raise RuntimeError("Max search depth exceeded")
            input, decorator = input.pop_decorator()
            if select(decorator):
                return decorator
            depth += 1

