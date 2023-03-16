from __future__ import annotations
from . import warning

from tango.core.profiler import ValueMeanProfiler
from tango.core.dataio import FormatDescriptor, AbstractInstruction

from abc import ABC, ABCMeta, abstractmethod
from typing import Union, Sequence, Iterable, Iterator, AsyncIterator, BinaryIO
from functools import partial, partialmethod
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

    def __init__(self):
        self.id = self.uniq_id

    @classmethod
    @property
    def uniq_id(cls) -> int:
        cls._COUNTER += 1
        return cls._COUNTER

    ## Abstract methods ##
    @property
    @abstractmethod
    def decorated(self) -> bool:
        pass

    @abstractmethod
    def ___iter___(self) -> Iterator[AbstractInstruction]:
        pass

    @abstractmethod
    async def ___aiter___(self) -> AsyncIterator[AbstractInstruction]:
        pass

    @abstractmethod
    def ___repr___(self) -> str:
        pass

    @abstractmethod
    def ___len___(self) -> Optional[int]:
        pass

    @abstractmethod
    def ___eq___(self, other: AbstractInput) -> bool:
        pass

    @abstractmethod
    def ___getitem___(self, idx: Union[int, slice]) -> AbstractInput:
        pass

    @abstractmethod
    def ___add___(self, other: AbstractInput) -> AbstractInput:
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
    async def ___aiter___(self) -> AsyncIterator[AbstractInstruction]:
        for e in iter(self):
            yield e

    def ___repr___(self) -> str:
        return f"{self.__class__.__name__}:0x{self.id:08X}"

    def ___len___(self) -> None:
        # if NotImplementedError is raised, then tuple(input) fails;
        # if NotImplemented is returned, tuple resorts to dynamic allocation
        return NotImplemented

    def __bool__(self) -> bool:
        # we define this so that __len__ is not used to test for truthiness
        return True

    def ___eq___(self, other: AbstractInput) -> bool:
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

    def ___getitem___(self, idx: Union[int, slice]) -> BaseInput:
        return SlicingDecorator(idx)(self)

    def ___add___(self, other: AbstractInput) -> BaseInput:
        return JoiningDecorator(suffix=(other,))(self)

    def flatten(self, inplace: bool=False) -> BaseInput:
        return MemoryCachingDecorator()(self, inplace=inplace)

    @property
    def decorated(self):
        return False

class EmptyInput(BaseInput):
    singleton: EmptyInput = None

    def __new__(cls):
        if not cls.singleton:
            cls.singleton = super().__new__(cls)
        return cls.singleton

    def ___iter___(self):
        return
        yield

    async def ___aiter___(self):
        return
        yield

    def ___repr___(self) -> str:
        return "EmptyInput"

    def ___len___(self) -> int:
        return 0

    def ___getitem___(self, idx: Union[int, slice]) -> EmptyInput:
        return self

    def ___add___(self, other: AbstractInput) -> AbstractInput:
        return other

class DecoratedInput(BaseInput):
    def __init__(self, decorator: AbstractDecorator, depth: int=0):
        super().__init__()
        self.___decorator___ = decorator
        self.___decorator_depth___ = depth

    def ___iter___(self):
        raise NotImplementedError

    def __del__(self):
        if self.decorated:
            self.___decorator___.undecorate()

    @property
    def decorated(self):
        return (self.___decorator_depth___ > 0) or self.___decorator___

    def pop_decorator(self) -> (AbstractInput, AbstractDecorator):
        decorator = self.___decorator___
        return decorator.pop(), decorator

    def search_decorator_stack(self, select: Callable[[AbstractDecorator], bool],
            *, max_depth: int=0) -> AbstractDecorator:
        depth = 0
        input = self
        while True:
            if not input.decorated:
                raise RuntimeError("Reached end of decorator stack")
            elif max_depth > 0 and depth >= max_depth:
                raise RuntimeError("Max search depth exceeded")
            input, decorator = input.pop_decorator()
            if select(decorator):
                return decorator
            depth += 1

class AbstractDecoratorMeta(ABCMeta):
    ALL_DECORATABLE_METHODS = {'___iter___', '___aiter___', '___len___',
        '___add___', '___getitem___', '___repr___', '___eq___'}

    def __new__(metacls, name, bases, namespace):
        if 'DECORATABLE_METHODS' not in namespace:
            kls_dec_methods = metacls.ALL_DECORATABLE_METHODS & namespace.keys()
            mro_dec_methods = kls_dec_methods.copy()
            for base in bases:
                if hasattr(base, 'DECORATABLE_METHODS'):
                    mro_dec_methods |= base.DECORATABLE_METHODS
            namespace['DECORATABLE_METHODS'] = mro_dec_methods
            namespace['OWN_DECORATABLE_METHODS'] = kls_dec_methods
        kls = super().__new__(metacls, name, bases, namespace)
        return kls

class AbstractDecorator(ABC, metaclass=AbstractDecoratorMeta):
    @abstractmethod
    def __call__(self, input: AbstractInput):
        return input

    @abstractmethod
    def pop(self) -> AbstractInput:
        raise NotImplementedError

class BaseDecorator(AbstractDecorator):
    def __call__(self, input, *, inplace=False, methods=None) -> AbstractInput:
        self._handle_copy(input, inplace=inplace)
        methods = methods or self.DECORATABLE_METHODS

        for name in methods:
            func = getattr(self, name)
            oldfunc = getattr(input, name, None)
            newfunc = partialmethod(partial(func), oldfunc).__get__(self._input)
            setattr(self._input, name, newfunc)

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

    def undecorate(self, *, methods=None):
        assert getattr(self, '_input', None) is not None, \
                "Decorator has not been called before!"
        methods = methods or self.DECORATABLE_METHODS
        for name in methods:
            newfunc = getattr(self._input, name, None)
            # extract `orig` from the partial function
            oldfunc = newfunc._partialmethod.args[0]
            setattr(self._input, name, oldfunc)
        self._input.___decorator___ = None
        self._input = None

    def pop(self) -> AbstractInput:
        if not self.DECORATABLE_METHODS or not hasattr(self, '_input'):
            # an "empty" decorator decorates no inputs
            return None
        # a decorator usually decorates at least one method, which allows us to
        # extract the `orig` argument and return its owner
        decorated_method = getattr(self._input, next(iter(self.OWN_DECORATABLE_METHODS)))
        # decorated methods are partial functions with args[0] == orig
        orig = decorated_method._partialmethod.args[0]
        owner = orig.__self__
        return owner

    def ___repr___(self, input, orig):
        return f"{self.__class__.__name__}({orig()})"

    @classmethod
    def get_parent_class(cls, method):
        """
        Gets the class that defined the method.
        Courtesy of: https://stackoverflow.com/a/25959545
        """
        if isinstance(method, partial):
            return cls.get_parent_class(method.func)
        if inspect.ismethod(method) or (inspect.isbuiltin(method) and \
                getattr(method, '__self__', None) is not None and \
                getattr(method.__self__, '__class__', None)):
            for cls in inspect.getmro(method.__self__.__class__):
                if method.__name__ in cls.__dict__:
                    return cls
            # fallback to __qualname__ parsing
            method = getattr(method, '__func__', method)
        # handle special descriptor objects
        return getattr(method, '__objclass__', None)

class MemoryCachingDecorator(AbstractDecorator):
    def __call__(self, input, *, inplace=False) -> AbstractInput:
        if not inplace or not input.decorated:
            new_input = DecoratedInput(self, depth=1)
        else:
            new_input = input

        new_input._cached_repr = repr(input)
        new_input._cached_iter = tuple(input)
        new_input._cached_len = len(new_input._cached_iter)

        cls = self.__class__
        for name in self.DECORATABLE_METHODS:
            func = getattr(cls, name)
            new_func = types.MethodType(func, new_input)
            setattr(new_input, name, new_func)

        self._input = new_input
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

class IterCachingDecorator(BaseDecorator):
    DECORATOR_MAX_DEPTH = 10

    def __call__(self, input, *, inplace=False, methods=None) -> AbstractInput:
        methods = methods or self.DECORATABLE_METHODS
        fn_name = '___iter___'
        filtered = methods - {fn_name}
        decorated = super().__call__(input, inplace=inplace, methods=filtered)
        if fn_name in methods:
            func = getattr(self, fn_name)
            olditer = getattr(input, fn_name, None)
            # We use a tee so that repeated accesses to the decorated input do
            # not result in repeated accesses to the original, which may have
            # unintended side effects.
            self._orig = olditer
            self._tee = self._orig()
            newfunc = partialmethod(
                partial(func), self.get_iter).__get__(self._input)
            setattr(decorated, fn_name, newfunc)

        if decorated.___decorator_depth___ > self.DECORATOR_MAX_DEPTH:
            return MemoryCachingDecorator()(decorated, inplace=inplace)
        else:
            return decorated

    def get_iter(self):
        self._tee, res = tee(self._tee, 2)
        return res

    def pop(self) -> AbstractInput:
        fn_name = '___iter___'
        if not self.DECORATABLE_METHODS or not hasattr(self, '_input') or \
                not fn_name in self.DECORATABLE_METHODS:
            return super().pop()

        decorated_method = getattr(self._input, fn_name)
        return self._orig.__self__

    def ___iter___(self, input, orig):
        return orig()

    def ___repr___(self, input, orig):
        return f"shielded({orig()})"

    @classmethod
    def shield(cls, input, inplace=True) -> AbstractInput:
        return cls()(input, inplace=inplace, methods={'___iter___'})

class SlicingDecorator(IterCachingDecorator):
    def __init__(self, idx):
        super().__init__()
        if isinstance(idx, slice):
            self._start = idx.start or 0
            self._stop = idx.stop
            self._step = idx.step or 1
        else:
            self._start = idx
            self._stop = idx + 1
            self._step = 1

    def __call__(self, input, **kwargs) -> AbstractInput:
        if self._start == 0 and self._stop == None and self._step == 1:
            return input
        elif input.decorated and \
                isinstance(input.___decorator___, self.__class__) and \
                input.___decorator___._step == self._step:
            input, other = input.pop_decorator()
            new_start = self._start + other._start
            if self._stop is not None:
                new_stop = new_start + (self._stop - self._start)
                assert other._stop is None or \
                    other._stop >= new_stop
            else:
                new_stop = other._stop
            self._start = new_start
            self._stop = new_stop

        return super().__call__(input, **kwargs)

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

class JoiningDecorator(IterCachingDecorator):
    def __init__(self, *,
            prefix: Sequence[AbstractInput]=(),
            suffix: Sequence[AbstractInput]=(), flatten_joined: bool=True):
        super().__init__()
        prefix = filter(lambda x: not isinstance(x, EmptyInput), prefix)
        suffix = filter(lambda x: not isinstance(x, EmptyInput), suffix)

        if flatten_joined:
            self._prefix = tuple(y
                for x in map(self._flatten_joined, prefix)
                    for y in x)
            self._suffix = tuple(y
                for x in map(self._flatten_joined, suffix)
                    for y in x)
        else:
            self._prefix = tuple(prefix)
            self._suffix = tuple(suffix)

    @classmethod
    def _flatten_joined(cls, inp: AbstractInput) -> Sequence[AbstractInput]:
        if inp.decorated and isinstance(inp.___decorator___, cls):
            inp, other = inp.pop_decorator()
            prefix = sum((cls._flatten_joined(x) for x in other._prefix),
                start=())
            suffix = sum((cls._flatten_joined(x) for x in other._suffix),
                start=())
            return (*prefix, inp, *suffix)
        else:
            return (inp,)

    def __call__(self, input, **kwargs) -> AbstractInput:
        if not self._prefix and not self._suffix:
            return input
        if input.decorated and isinstance(input.___decorator___, self.__class__):
            input, other = input.pop_decorator()
            self._prefix = self._prefix + other._prefix
            self._suffix = other._suffix + self._suffix
        return super().__call__(input, **kwargs)

    def ___iter___(self, input, orig):
        return chain(*self._prefix, orig(), *self._suffix)

    def ___repr___(self, input, orig):
        id = f'0x{self._input_id:08X}'
        pres = (f'0x{x.id:08X}' for x in self._prefix)
        sufs = (f'0x{x.id:08X}' for x in self._suffix)
        return f'JoinedInput:0x{input.id:08X} ({" || ".join((*pres, id, *sufs))})'

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
