from . import warning
from abc import ABCMeta
from input import PreparedInput
from interaction import InteractionBase
from abc import abstractmethod
from dataclasses import dataclass
from functools import wraps, partial
from io import RawIOBase
from typing import Sequence, Iterable, BinaryIO
import os

@dataclass(frozen=True)
class FormatDescriptor:
    """
    This is just a helper class to be used when passing around
    information describing the format of the serialized data.
    Additional fields could be added by inheritance, in case a
    format can be further specialized, depending on the channel
    being used.
    """
    typ: str

class SerializedMetaInput(ABCMeta):
    def __new__(metacls, name, bases, namespace, *, typ):
        # we add SerializedInput as the last base class in the MRO
        bases = bases + (SerializedInput,)
        kls = super().__new__(metacls, name, bases, namespace)
        return Serializer(typ)(kls)

class SerializedInput(PreparedInput):
    def __init__(self, *, file: str | BinaryIO, fmt: FormatDescriptor,
            load: bool=False, **kwargs):
        super().__init__(**kwargs)
        if isinstance(file, RawIOBase):
            self._file = file
        elif isinstance(file, str):
            self._file = open(file, mode='ab+')
            self._file.seek(0, os.SEEK_SET)
        else:
            raise TypeError("`file` must either be an open binary stream,"
                            " or a string containing the path to the file.")

        self._fmt = fmt

        if load:
            self.load()

    def __del__(self):
        if not self._file.closed:
            self._file.close()

    def load(self):
        if self._file.closed:
            warning(f"Attempted to load from already closed stream {self._file}.")
            return
        self.extend(self.loadi())
        self._file.close()

    def dump(self, itr: Iterable[InteractionBase]=None, /):
        if self._file.closed:
            warning(f"Attempted to dump to already closed stream {self._file}.")
            return
        if itr:
            self.extend(itr)
        self.dumpi(itr or self._interactions)
        self._file.close()

    @abstractmethod
    def loadi(self) -> Iterable[InteractionBase]:
        pass

    @abstractmethod
    def dumpi(self, itr: Iterable[InteractionBase], /):
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
