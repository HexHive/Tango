from __future__ import annotations
from . import warning

from tango.core.profiler import ValueMeanProfiler
from tango.core.dataio import FormatDescriptor, AbstractInstruction

from abc import ABC, ABCMeta, abstractmethod
from typing import Union, Sequence, Iterable, Iterator, AsyncIterator, BinaryIO
from functools import partial, partialmethod, cached_property, reduce
from itertools import islice, chain, tee
import inspect
import types
import struct
import io
import os

__all__ = [
    'AbstractInput', 'BaseInput', 'AbstractDecorator', 'BaseDecorator',
    'IterCachingDecorator', 'SlicingDecorator', 'JoiningDecorator',
    'MemoryCachingDecorator', 'PreparedInput', 'SerializedInput',
    'SerializedInputMeta', 'Serializer', 'EmptyInput'
]

class AbstractInput(ABC):
    _COUNTER = 0

    def __new__(cls, **kwargs):
        return super().__new__(cls)

    def __init__(self, **kwargs):
        self.id = self.uniq_id

    @classmethod
    @property
    def uniq_id(cls) -> int:
        AbstractInput._COUNTER += 1
        return AbstractInput._COUNTER

    ## Abstract methods ##
    @property
    @abstractmethod
    def decorated(self) -> bool:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[AbstractInstruction]:
        pass

    @abstractmethod
    async def __aiter__(self) -> AsyncIterator[AbstractInstruction]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __len__(self) -> Optional[int]:
        pass

    @abstractmethod
    def __eq__(self, other: AbstractInput) -> bool:
        pass

    @abstractmethod
    def __getitem__(self, idx: Union[int, slice]) -> AbstractInput:
        pass

    @abstractmethod
    def __add__(self, other: AbstractInput) -> AbstractInput:
        pass

    @abstractmethod
    def __hash__(self):
        pass

class BaseInput(AbstractInput):
    async def __aiter__(self) -> AsyncIterator[AbstractInstruction]:
        for e in iter(self):
            yield e

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:0x{self.id:08X}"

    def __len__(self) -> None:
        return NotImplemented

    def __bool__(self) -> bool:
        # we define this so that __len__ is not used to test for truthiness
        return True

    def __eq__(self, other: AbstractInput) -> bool:
        try:
            return all(type(x) == type(y) and x == y
                for x, y in zip(iter(self), iter(other), strict=True))
        except ValueError:
            return False

    def __getitem__(self, idx: Union[int, slice]) -> BaseInput:
        return SlicingDecorator(self, idx=idx)

    def __add__(self, other: AbstractInput) -> BaseInput:
        return JoiningDecorator(self, suffix=(other,))

    @cached_property
    def _hash(self):
        return reduce(lambda v, e: hash(e) ^ (v >> 1), self, 0)

    def __hash__(self):
        return self._hash

    def flatten(self) -> BaseInput:
        return MemoryCachingDecorator(self)

    def shield(self) -> BaseInpit:
        return IterCachingDecorator(self)

    @property
    def decorated(self):
        return False

class EmptyInputMeta(ABCMeta):
    singleton: EmptyInput = None
    def __call__(cls):
        if cls.singleton is not None:
            return cls.singleton
        obj = super().__call__()
        cls.singleton = obj
        return obj

class EmptyInput(BaseInput, metaclass=EmptyInputMeta):
    def __iter__(self):
        return
        yield

    async def __aiter__(self):
        return
        yield

    def __repr__(self) -> str:
        return "EmptyInput"

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: Union[int, slice]) -> EmptyInput:
        return self

    def __add__(self, other: AbstractInput) -> AbstractInput:
        return other

class DecoratedFunctionType:
    def __init__(self, function: types.FunctionType, /):
        self.func = function

    def __get__(self, obj, owner):
        if obj is not None:
            orig_fn = getattr(obj._orig, self.func.__name__)
            meth = partialmethod(self.func, orig=orig_fn).__get__(obj, owner)
        else:
            meth = self.func.__get__(obj, owner)
        return meth

class AbstractDecoratorMeta(ABCMeta):
    DECORATABLE_METHODS = {'__iter__', '__aiter__', '__len__',
        '__add__', '__getitem__', '__repr__', '__eq__'}
    def __new__(metacls, name, bases, namespace, *, desc=DecoratedFunctionType):
        for k, v in namespace.items():
            if k not in metacls.DECORATABLE_METHODS:
                continue
            namespace[k] = desc(v)
        typ = super().__new__(metacls, name, bases, namespace)
        return typ

    def __call__(cls, input: AbstractInput, /, **kwargs):
        obj = cls.__new__(cls, input, **kwargs)
        if not getattr(obj, 'initialized', False): # input was not initialized
            obj.__init__(input, **kwargs)
        return obj

class AbstractDecorator(AbstractInput, metaclass=AbstractDecoratorMeta):
    initialized: bool = False
    def __new__(cls, input: AbstractInput, /, **kwargs):
        obj = super().__new__(cls, **kwargs)
        obj._orig = input
        return obj

    def __init__(self, input: AbstractInput, /, **kwargs):
        super().__init__(**kwargs)
        self.initialized = True

    def __getattr__(self, name):
        return getattr(self._orig, name)

identity = lambda x: x
class MemoryCachingDecorator(AbstractDecorator, BaseInput, desc=identity):
    _depth: int = 0
    def __new__(cls, input: AbstractInput, /, **kwargs):
        if isinstance(input, cls):
            return input
        obj = super().__new__(cls, input, **kwargs)
        obj._cached_repr = repr(input)
        obj._cached_iter = tuple(input)
        obj._cached_len = len(obj._cached_iter)
        obj._cached_hash = hash(input)
        obj.id = input.id

        if (hook := getattr(input, '__cache_hook__', None)):
            hook(obj)
        return obj

    def __init__(self, input: AbstractInput, /, **kwargs):
        del self._orig
        # we skip initialization of parents
        self.initialized = True

    @property
    def decorated(self):
        return False

    def __iter__(self):
        return iter(self._cached_iter)

    def __repr__(self):
        return self._cached_repr

    def __len__(self):
        return self._cached_len

    def __getattr__(self, name):
        raise AttributeError

    def __hash__(self):
        return self._cached_hash

class BaseDecorator(AbstractDecorator, BaseInput):
    def __new__(cls, input: AbstractInput, /, **kwargs):
        depth = getattr(input, '_depth', 0)
        obj = super().__new__(cls, input, **kwargs)
        obj._depth = depth + 1
        ValueMeanProfiler("decorator_depth", samples=100)(obj._depth)
        return obj

    @property
    def decorated(self):
        return True

    def pop(self) -> AbstractInput:
        return self._orig

    def search_decorator_stack(self, select: Callable[[AbstractDecorator], bool],
            *, max_depth: int=0) -> AbstractDecorator:
        depth = 0
        input = self
        while True:
            if not input.decorated:
                raise RuntimeError("Reached end of decorator stack")
            elif max_depth > 0 and depth >= max_depth:
                raise RuntimeError("Max search depth exceeded")
            if select(input):
                return input
            input = input.pop()
            depth += 1

    def __repr__(self, *, orig):
        return f"{self.__class__.__name__}({orig()})"

class TeeCachedFunctionType(DecoratedFunctionType):
    def __new__(cls, function: types.FunctionType, /):
        if function.__name__ != '__iter__':
            obj = super().__new__(DecoratedFunctionType)
            obj.__init__(function)
            return obj
        else:
            return super().__new__(cls)

    def __init__(self, function: types.FunctionType, /):
        super().__init__(function)

    def __get__(self, obj, owner):
        if obj is not None:
            if not obj._tee:
                obj._tee = getattr(obj._orig, self.func.__name__)()
            obj._tee, cpy = tee(obj._tee, 2)
            meth = partialmethod(lambda _: cpy).__get__(obj, owner)
        else:
            meth = super().__get__(obj, owner)
        return meth

class IterCachingDecorator(BaseDecorator, desc=TeeCachedFunctionType):
    _tee: Optional[tee] = None

    def __repr__(self, *, orig):
        return f"shielded({orig()})"

    def __iter__(self):
        # this exists only to signal to the metaclass to generate the proper
        # field descriptor for __iter__ that wraps the original
        return
        yield

class SlicingDecorator(BaseDecorator):
    def __new__(cls, input: AbstractInput, /, *, idx: slice | int, **kwargs):
        if isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop
            step = idx.step or 1
        else:
            start = idx
            stop = idx + 1
            step = 1

        if start == 0 and stop == None and step == 1:
            return input.shield()

        obj = super().__new__(cls, input, **kwargs)
        if input.decorated and \
                isinstance(input, cls) and \
                input._step == step:
            new_start = start + input._start
            if stop is not None:
                new_stop = new_start + (stop - start)
                assert input._stop is None or \
                    input._stop >= new_stop
            else:
                new_stop = input._stop
            start = new_start
            stop = new_stop
            obj._orig = input._orig

        obj._start = start
        obj._stop = stop
        obj._step = step
        return obj

    def __init__(self, input: AbstractInput, /, *, idx: slice | int, **kwargs):
        super().__init__(input, **kwargs)

    def __iter__(self, *, orig):
        return islice(orig(), self._start, self._stop, self._step)

    def __repr__(self, *, orig):
        if self._stop == self._start + 1 and self._step == 1:
            fmt = f'{self._start}'
        elif self._stop is None:
            if self._step == 1:
                fmt = f'{self._start}:'
            else:
                fmt = f'{self._start}::{self._step}'
        else:
            fmt = f'{self._start}:{self._stop}:{self._step}'
        return f'SlicedInput:0x{self.id:08X} (0x{self._orig.id:08X}[{fmt}])'

class JoiningDecorator(BaseDecorator):
    def __new__(cls, input: AbstractInput, /, *, flatten_joined: bool=True,
            prefix: Sequence[AbstractInput]=(),
            suffix: Sequence[AbstractInput]=(), **kwargs):
        prefix = filter(lambda x: not isinstance(x, EmptyInput), prefix)
        suffix = filter(lambda x: not isinstance(x, EmptyInput), suffix)

        if flatten_joined:
            prefix = tuple(y
                for x in map(cls._flatten_joined, prefix)
                    for y in x)
            suffix = tuple(y
                for x in map(cls._flatten_joined, suffix)
                    for y in x)
        else:
            prefix = tuple(prefix)
            suffix = tuple(suffix)

        if not prefix and not suffix:
            return input.shield()

        obj = super().__new__(cls, input, **kwargs)
        if input.decorated and isinstance(input, cls):
            prefix = prefix + input._prefix
            suffix = input._suffix + suffix
            obj._orig = input._orig

        obj._prefix = prefix
        obj._suffix = suffix
        return obj


    def __init__(self, input: AbstractInput, /, *, flatten_joined: bool=True,
            prefix: Sequence[AbstractInput]=(),
            suffix: Sequence[AbstractInput]=(), **kwargs):
        super().__init__(input, **kwargs)

    @classmethod
    def _flatten_joined(cls, inp: AbstractInput) -> Sequence[AbstractInput]:
        if inp.decorated and isinstance(inp, cls):
            prefix = sum((cls._flatten_joined(x) for x in inp._prefix),
                start=())
            suffix = sum((cls._flatten_joined(x) for x in inp._suffix),
                start=())
            return (*prefix, inp._orig, *suffix)
        else:
            return (inp,)

    def __iter__(self, *, orig):
        return chain(*self._prefix, orig(), *self._suffix)

    def __repr__(self, *, orig):
        id = f'0x{self._orig.id:08X}'
        pres = (f'0x{x.id:08X}' for x in self._prefix)
        sufs = (f'0x{x.id:08X}' for x in self._suffix)
        return f'JoinedInput:0x{self.id:08X} ({" || ".join((*pres, id, *sufs))})'

class PreparedInput(BaseInput):
    """
    A buffered input. All instructions are readily available and can be exported
    to a file.
    """
    def __init__(self, *, instructions: Sequence[AbstractInstruction]=None,
            **kwargs):
        super().__init__(**kwargs)
        self._instructions = []
        if instructions:
            self._instructions.extend(instructions)

    def append(self, instruction: AbstractInstruction):
        self._instructions.append(instruction)

    def extend(self, instructions: Sequence[AbstractInstruction]):
        self._instructions.extend(instructions)

    def __iter__(self):
        return iter(self._instructions)

    def __len__(self):
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
            warning("Attempted to load from already closed stream %s.",
                self._file)
            return
        if self._read_magic():
            name = self._read_long_name()
            self._name = f'{self._name}::{name}'
        self.extend(self.loadi())
        self._file.close()
        return self

    def dump(self, itr: Iterable[AbstractInstruction]=None, /, *, name: str=None):
        if self._file.closed:
            warning("Attempted to dump to already closed stream %s.",
                self._file)
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
            raise ValueError(f"Cannot serialize class {kls}."
                " Format '{typ}' is already mapped to {dk}!")
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
