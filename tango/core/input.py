from __future__ import annotations
from . import warning

from tango.core.profiler import ValueMeanProfiler
from tango.core.dataio import FormatDescriptor, AbstractInstruction

from abc import ABC, ABCMeta, abstractmethod
from typing import Union, Sequence, Iterable, BinaryIO
from functools import partial, partialmethod
from itertools import islice, chain
import inspect
import types
import struct
import io
import os

__all__ = [
    'AbstractInput', 'BaseInput', 'BaseDecorator', 'SlicingDecorator',
    'JoiningDecorator', 'MemoryCachingDecorator', 'PreparedInput',
    'SerializedInput', 'SerializedInputMeta', 'Serializer'
]

class AbstractInput(ABC):
    _COUNTER = 0

    def __init__(self):
        self.id = self.uniq_id

    @classmethod
    @property
    def uniq_id(cls):
        cls._COUNTER += 1
        return cls._COUNTER

    ## Abstract methods ##
    @property
    @abstractmethod
    def decorated(self):
        pass

    @abstractmethod
    def ___iter___(self):
        pass

    @abstractmethod
    async def ___aiter___(self):
        pass

    @abstractmethod
    def ___repr___(self):
        pass

    @abstractmethod
    def ___len___(self):
        pass

    @abstractmethod
    def ___eq___(self, other: AbstractInput):
        pass

    @abstractmethod
    def ___getitem___(self, idx: Union[int, slice]):
        pass

    @abstractmethod
    def ___add___(self, other: AbstractInput):
        pass

    ## Decoratable functions ##
    # These definitions are needed so that a decorator can override the behavior
    # of "special methods". The python interpreter does not resolve special
    # methods as it would other class attributes, so a level of indirection is
    # needed to bypass this limitation.
    def __iter__(self):
        return self.___iter___()

    def __aiter__(self):
        return self.___aiter___()

    def __repr__(self):
        return self.___repr___()

    def __len__(self):
        return self.___len___()

    def __eq__(self, other):
        return self.___eq___(other)

    def __getitem__(self, idx):
        return self.___getitem___(idx)

    def __add__(self, other):
        return self.___add___(other)

class BaseInput(AbstractInput):
    async def ___aiter___(self):
        for e in iter(self):
            yield e

    def ___repr___(self):
        return f"{self.__class__.__name__}:0x{self.id:08X}"

    def ___len___(self):
        # if NotImplementedError is raised, then tuple(input) fails;
        # if None is returned, tuple resorts to dynamic allocation
        return None

    def __bool__(self):
        # we define this so that __len__ is not used to test for truthiness
        return True

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

    def ___add___(self, other: AbstractInput):
        return JoiningDecorator(other)(self)

    def flatten(self, inplace: bool=False):
        return MemoryCachingDecorator()(self, inplace=inplace)

    @property
    def decorated(self):
        return False

class BaseDecoratorMeta(type):
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

class BaseDecorator(metaclass=BaseDecoratorMeta):
    DECORATOR_MAX_DEPTH = 10

    def __call__(self, input, inplace=False) -> AbstractInput:
        self._handle_copy(input, inplace=inplace)

        for name in self.DECORATABLE_METHODS:
            func = getattr(self, name)
            oldfunc = getattr(input, name, None)
            newfunc = partialmethod(partial(func), oldfunc).__get__(self._input)
            setattr(self._input, name, newfunc)

        if self._input.___decorator_depth___ > self.DECORATOR_MAX_DEPTH:
            return MemoryCachingDecorator()(self._input, inplace=inplace)
        else:
            return self._input

    def _handle_copy(self, input, *, inplace) -> AbstractInput:
        self._input_id = input.id
        if input.decorated:
            depth = input.___decorator_depth___ + 1
        else:
            depth = 1
        if not inplace or not input.decorated:
            self._input = DecoratedInput(self, depth)
        else:
            self._input = input
        ValueMeanProfiler("decorator_depth", samples=100)(depth)
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

    def pop(self) -> AbstractInput:
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

class SlicingDecorator(BaseDecorator):
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

    def __call__(self, input, inplace=False): # -> AbstractInput:
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

        return super().__call__(input, inplace=inplace)

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

class JoiningDecorator(BaseDecorator):
    def __init__(self, *others):
        self._others = list(others)

    def __call__(self, input, inplace=False): # -> AbstractInput:
        if input.decorated and isinstance(input.___decorator___, self.__class__):
            input, other = input.pop_decorator()
            self._others = other._others + self._others
        return super().__call__(input, inplace=inplace)

    def ___iter___(self, input, orig):
        return chain(orig(), *self._others)

    def ___repr___(self, input, orig):
        id = f'0x{self._input_id:08X}'
        ids = (f'0x{x.id:08X}' for x in self._others)
        return f'JoinedInput:0x{input.id:08X} ({" || ".join((id, *ids))})'

class MemoryCachingDecorator(BaseDecorator):
    def __call__(self, input, inplace=False): # -> AbstractInput:
        new_input = self._handle_copy(input, inplace=inplace)

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

    def pop(self) -> AbstractInput:
        raise NotImplementedError("This should never be reachable!")

    def ___iter___(self):
        return iter(self._cached_iter)

    def ___repr___(self):
        return self._cached_repr

    def ___len___(self):
        return self._cached_len

class DecoratedInput(BaseInput):
    def __init__(self, decorator: BaseDecorator, depth: int=0):
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

    def pop_decorator(self) -> (AbstractInput, BaseDecorator):
        decorator = self.___decorator___
        return decorator.pop(), decorator

    def search_decorator_stack(self, select: Callable[[BaseDecorator], bool], max_depth: int=None) -> BaseDecorator:
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

class PreparedInput(BaseInput):
    """
    A buffered input. All instructions are readily available and can be exported
    to a file.
    """
    def __init__(self, *, instructions: Sequence[AbstractInstruction]=None, **kwargs):
        super().__init__(**kwargs)
        self._instructions = []
        if instructions:
            self._instructions.extend(instructions)

    def append(self, instruction: AbstractInstruction):
        self._instructions.append(instruction)

    def extend(self, instructions: Sequence[AbstractInstruction]):
        self._instructions.extend(instructions)

    def ___iter___(self):
        return iter(self._instructions)

    def ___len___(self):
        return len(self._instructions)

class SerializedInputMeta(ABCMeta):
    def __new__(metacls, name, bases, namespace, *, typ):
        for base in bases:
            if issubclass(base, SerializedInput):
                break
        else:
            # we add SerializedInput as the last base class in the MRO
            bases = bases + (SerializedInput,)
        kls = super().__new__(metacls, name, bases, namespace)
        return Serializer(typ)(kls)

class SerializedInput(PreparedInput):
    MAGIC = b'SERL\0'

    def __init__(self, *, file: str | BinaryIO, fmt: FormatDescriptor,
            load: bool=False, **kwargs):
        super().__init__(**kwargs)
        if isinstance(file, io.RawIOBase):
            self._name = repr(file)
            self._file = file
        elif isinstance(file, str):
            mode = 'rb+' if os.path.isfile(file) else 'wb+'
            self._name = file
            self._file = open(file, mode=mode)
            self._file.seek(0, io.SEEK_SET)
        else:
            raise TypeError("`file` must either be an open binary stream,"
                            " or a string containing the path to the file.")

        self._fmt = fmt

        if load:
            self.load()

    def __del__(self):
        if hasattr(self, '_file') and not self._file.closed:
            self._file.close()

    def _read_magic(self):
        data = self._file.read(len(self.MAGIC))
        if data == self.MAGIC:
            return True
        self._file.seek(-len(data), io.SEEK_CUR)
        return False

    def _write_magic(self):
        self._file.write(self.MAGIC)

    def _read_long_name(self) -> str:
        len_fmt = 'I'
        name_len = self._file.read(struct.calcsize(len_fmt))
        name_len, = struct.unpack(len_fmt, name_len)
        name_fmt = f'{name_len}s'
        name = self._file.read(struct.calcsize(name_fmt))
        name, = struct.unpack(name_fmt, name)
        return name.decode()

    def _write_long_name(self, name):
        if isinstance(name, str):
            name = name.encode()
        self._file.write(struct.pack('I', len(name)))
        self._file.write(struct.pack(f'{len(name)}s', name))

    def load(self) -> SerializedInput:
        if self._file.closed:
            warning(f"Attempted to load from already closed stream {self._file}.")
            return
        if self._read_magic():
            name = self._read_long_name()
            self._name = f'{self._name}::{name}'
        self.extend(self.loadi())
        self._file.close()
        return self

    def dump(self, itr: Iterable[AbstractInstruction]=None, /, *, name: str=None):
        if self._file.closed:
            warning(f"Attempted to dump to already closed stream {self._file}.")
            return
        if itr:
            self.extend(itr)
        if name:
            self._write_magic()
            self._write_long_name(name)
            self._name = f'{self._name}::{name}'
        self.dumpi(itr or self._instructions)
        os.ftruncate(self._file.fileno(), self._file.tell())
        self._file.close()

    @abstractmethod
    def loadi(self) -> Iterable[AbstractInstruction]:
        pass

    @abstractmethod
    def dumpi(self, itr: Iterable[AbstractInstruction], /):
        pass

class Serializer:
    mapping: dict = {}
    get = staticmethod(mapping.get)

    def __init__(self, typ: str, /):
        self._typ = typ

    def __call__(self, kls: type):
        return self.serialize(kls, self._typ)

    @classmethod
    def serialize(cls, kls: type, typ: str):
        if (dk := cls.mapping.get(typ)) is not None:
            raise ValueError(f"Cannot serialize class {kls}. Format '{typ}' is already mapped to {dk}!")
        cls.mapping[typ] = kls
        return kls

    @classmethod
    def __getitem__(cls, fmt: FormatDescriptor, /):
        kls = cls.mapping[fmt.typ]
        return partial(kls, fmt=fmt)

    @classmethod
    def get(cls, fmt: FormatDescriptor, /, *args):
        kls = cls.mapping.get(fmt.typ)
        if kls is not None:
            return partial(kls, fmt=fmt)
        elif args:
            return args[0]
