from . import debug, info

import sys
import mmap
import ctypes
import posix_ipc
from string import ascii_letters, digits

class ZoomLocation(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]

class ZoomFeedback(ctypes.Structure):
    NUMCARDS = 6
    NUMWEAPONS = 9
    NUMAMMO = 4

    _fields_ = [
        ("player_location", ZoomLocation),
        ("player_angle", ctypes.c_float),
        ("player_state", ctypes.c_int),
        ("health", ctypes.c_int),
        ("armor_points", ctypes.c_int), # c_int instead of bool because boolean is typedef'd as enum in doomtype.h
        ("cards", ctypes.c_int * NUMCARDS),
        ("weapon_owned", ctypes.c_int * NUMWEAPONS),
        ("ammo", ctypes.c_int * NUMAMMO),
        ("attacker_valid", ctypes.c_int), # c_int instead of bool because boolean is typedef'd as enum in doomtype.h
        ("attacker_location", ZoomLocation),
        ("did_secret", ctypes.c_int), # c_int instead of bool because boolean is typedef'd as enum in doomtype.h
        ("can_activate", ctypes.c_int), # c_int instead of bool because boolean is typedef'd as enum in doomtype.h
        ("tic_rate", ctypes.c_float),
        ("floor_is_lava", ctypes.c_int), # c_int instead of bool because boolean is typedef'd as enum in doomtype.h
        ("secret_sector", ctypes.c_int), # c_int instead of bool because boolean is typedef'd as enum in doomtype.h
        ("pickup_valid", ctypes.c_int), # c_int instead of bool because boolean is typedef'd as enum in doomtype.h
        ("pickup_type", ctypes.c_int),
        ("pickup_location", ZoomLocation),
        ("level_finished", ctypes.c_int), # c_int instead of bool because boolean is typedef'd as enum in doomtype.h
    ]

class ZoomStateReader:

    TANGOFUZZ_FEEDBACK_SHM = "/tangofuzz_feedback"

    valid_chars = frozenset("-_. %s%s" % (ascii_letters, digits))

    typecode_to_type = {
        'c': ctypes.c_char, 'u': ctypes.c_wchar,
        'b': ctypes.c_byte, 'B': ctypes.c_ubyte,
        'h': ctypes.c_short, 'H': ctypes.c_ushort,
        'i': ctypes.c_int, 'I': ctypes.c_uint,
        'l': ctypes.c_long, 'L': ctypes.c_ulong,
        'f': ctypes.c_float, 'd': ctypes.c_double
    }

    def __init__(self, create=False, force=False):
        # default vals so __del__ doesn't fail if __init__ fails to complete
        self._mem = None
        self._map = None
        self._owner = create

        _type = ZoomFeedback
        _size = ctypes.sizeof(_type)

        self._mem, self._map = self.init_mem(self.TANGOFUZZ_FEEDBACK_SHM, _type, _size, create, force)
        self._struct = _type.from_address(self.address_of_buffer(self._map))

    @staticmethod
    def init_mem(tag, typ, size, create, force):
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
    def struct(self):
        return self._struct

    @property
    def address(self):
        return self._map

    def __del__(self):
        if self._map is not None:
            self._map.close()
        if self._mem is not None and self._owner:
            self._mem.unlink()
