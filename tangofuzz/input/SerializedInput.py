from __future__ import annotations
from . import warning
from abc import ABCMeta
from input import PreparedInput
from interaction import InteractionBase
from abc import abstractmethod
from dataclasses import dataclass
from functools import wraps, partial
from io import RawIOBase
from typing import Sequence, Iterable, BinaryIO
import struct
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
    MAGIC = b'SERL\0'

    def __init__(self, *, file: str | BinaryIO, fmt: FormatDescriptor,
            load: bool=False, **kwargs):
        super().__init__(**kwargs)
        if isinstance(file, RawIOBase):
            self._name = repr(file)
            self._file = file
        elif isinstance(file, str):
            mode = 'rb+' if os.path.isfile(file) else 'wb+'
            self._name = file
            self._file = open(file, mode=mode)
            self._file.seek(0, os.SEEK_SET)
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
        if self._file.read(len(self.MAGIC)) == self.MAGIC:
            return True
        self._file.seek(-len(self.MAGIC), os.SEEK_CUR)
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

    def dump(self, itr: Iterable[InteractionBase]=None, /, *, name: str=None):
        if self._file.closed:
            warning(f"Attempted to dump to already closed stream {self._file}.")
            return
        if itr:
            self.extend(itr)
        if name:
            self._write_magic()
            self._write_long_name(name)
            self._name = f'{self._name}::{name}'
        self.dumpi(itr or self._interactions)
        os.ftruncate(self._file.fileno(), self._file.tell())
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