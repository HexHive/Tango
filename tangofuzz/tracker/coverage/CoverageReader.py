from . import debug, info

import sys
import mmap
import ctypes
import posix_ipc
from string import ascii_letters, digits
import numpy as np

class CoverageReader:
    valid_chars = frozenset("-_. %s%s" % (ascii_letters, digits))

    typecode_to_type = {
        'c': ctypes.c_char, 'u': ctypes.c_wchar,
        'b': ctypes.c_byte, 'B': ctypes.c_ubyte,
        'h': ctypes.c_short, 'H': ctypes.c_ushort,
        'i': ctypes.c_int, 'I': ctypes.c_uint,
        'l': ctypes.c_long, 'L': ctypes.c_ulong,
        'f': ctypes.c_float, 'd': ctypes.c_double
    }

    def __init__(self, tag, size_tag, create=False, force=False):
        # default vals so __del__ doesn't fail if __init__ fails to complete
        self._mem = None
        self._map = None
        self._owner = create

        size_tag = self.ensure_tag(size_tag)

        # get the size of the coverage array
        _type = self.typecode_to_type['I']
        _size = ctypes.sizeof(_type)
        # FIXME this is prone to race conditions when launching multiple fuzzers
        _mem, _map = self.init_array(size_tag, _type, _size, False, False)
        self._length = _type.from_address(self.address_of_buffer(_map)).value
        _map.close()
        _mem.unlink()

        info(f"Obtained coverage map {self._length=}")

        tag = self.ensure_tag(tag)

        _type = self.typecode_to_type['B'] * self._length
        _size = ctypes.sizeof(_type)

        self._mem, self._map = self.init_array(tag, _type, _size, create, force)
        self._array = _type.from_address(self.address_of_buffer(self._map))

    @classmethod
    def ensure_tag(cls, tag):
        assert frozenset(tag[1:]).issubset(cls.valid_chars)
        if tag[0] != "/":
            tag = "/%s" % (tag,)
        return tag

    @staticmethod
    def init_array(tag, typ, size, create, force):
        # assert 0 <= size < sys.maxint
        assert 0 <= size < sys.maxsize
        flag = (0, posix_ipc.O_CREX)[create]
        try:
            _mem = posix_ipc.SharedMemory(tag, flags=flag, size=size)
        except posix_ipc.ExistentialError:
            if force:
                posix_ipc.unlink_shared_memory(tag)
                _mem = posix_ipc.SharedMemory(tag, flags=flag, size=size)
            else:
                raise

        _map = mmap.mmap(_mem.fd, _mem.size)
        _mem.close_fd()

        return _mem, _map

    @staticmethod
    def address_of_buffer(buf):
        return ctypes.addressof(ctypes.c_char.from_buffer(buf))

    @property
    def array(self):
        return self._array

    @property
    def length(self):
        return self._length

    @property
    def address(self):
        return self._map

    def __del__(self):
        if self._map is not None:
            self._map.close()
        if self._mem is not None and self._owner:
            self._mem.unlink()