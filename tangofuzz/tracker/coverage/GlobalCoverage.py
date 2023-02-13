from __future__   import annotations
from abc import ABC, ABCMeta, abstractmethod
from typing       import Sequence
from collections  import OrderedDict
from functools    import cache
from ctypes       import pythonapi, memset, memmove, addressof
from ctypes       import (POINTER as P,
                          c_bool as B,
                          c_ubyte as b,
                          c_uint32 as I,
                          c_size_t as S,
                          c_void_p as V,
                          cast as C,
                          byref)
import numpy as np

pythonapi.memcmp.argtypes = (V, V, S)

class GlobalCoverage(ABC):
    def __new__(cls, *args, **kwargs):
        if (bind_lib := kwargs.get('bind_lib')):
            return super(GlobalCoverage, CGlobalCoverage).__new__(CGlobalCoverage)
        else:
            return super(GlobalCoverage, NPGlobalCoverage).__new__(NPGlobalCoverage)

    def __init__(self, length: int, **kwargs):
        self._length = length

    @abstractmethod
    def update(self, coverage_map: Sequence) -> (Sequence, int, int):
        raise NotImplementedError

    @abstractmethod
    def revert(self, set_map: Sequence, map_hash: int):
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> GlobalCoverage:
        raise NotImplementedError

    @abstractmethod
    def copy_from(self, other: GlobalCoverage):
        if self._length != other._length:
            raise RuntimeError("Mismatching coverage map sizes")

    @abstractmethod
    def clear(self):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        return isinstance(other, GlobalCoverage)

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError

class CGlobalCoverage(GlobalCoverage):
    def __init__(self, *args, bind_lib, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_arr = (b * self._length)()
        self.clear()
        self._hash = 0

        self._bind_lib = bind_lib
        self._bind_lib.diff.argtypes = (
            P(b), # global coverage array (set_arr)
            P(b), # local coverage array (coverage_map)
            P(b), # pre-allocated set_map output buffer
            S, # common size of the coverage buffers
            P(I), # pre-allocated set_count output buffer
            P(I), # pre-allocated hash output buffer
            B, # boolean: apply in-place?
        )
        self._bind_lib.diff.restype = B # success?

        self._bind_lib.apply.argtypes = (
            P(b), # global coverage array (set_arr)
            P(b), # set_map buffer
            S # common size of the coverage buffers
        )
        self._bind_lib.apply.restype = B # success?

    def update(self, coverage_map: Sequence) -> (Sequence, int, int):
        set_map = (b * self._length)()
        set_count = I()
        map_hash = I()
        res = self._bind_lib.diff(self._set_arr, coverage_map, set_map, self._length,
            byref(set_count), byref(map_hash),
            True
        )
        self._hash ^= map_hash.value
        return set_map, set_count.value, map_hash.value

    def revert(self, set_map: Sequence, map_hash: int):
        self._bind_lib.apply(self._set_arr, set_map, self._length)
        self._hash ^= map_hash

    def clone(self) -> CGlobalCoverage:
        cpy = self.__class__(self._length, bind_lib=self._bind_lib)
        cpy.copy_from(self)
        return cpy

    def copy_from(self, other: CGlobalCoverage):
        super().copy_from(other)
        memmove(self._set_arr, other._set_arr, self._length)
        other._hash = self._hash

    def clear(self):
        memset(self._set_arr, 0, self._length)

    def __eq__(self, other):
        return super().__eq__(other) and \
            self._hash == other._hash and \
            pythonapi.memcmp(self._set_arr, other._set_arr, self._length) == 0

    def __hash__(self):
        return self._hash

class NPGlobalCoverage(GlobalCoverage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_arr = np.zeros(self._length, dtype=np.uint8)
        self.clear()

    def update(self, coverage_map: Sequence) -> (Sequence, int, int):
        coverage_map = np.ctypeslib.as_array(coverage_map)
        kls_arr = CLASS_LUT[coverage_map]
        set_map = (self._set_arr | kls_arr) ^ self._set_arr
        self._set_arr ^= set_map
        set_count = np.sum(HAMMING_LUT[set_map])
        map_hash = hash(set_map.data.tobytes())

        return set_map, set_count, map_hash

    def revert(self, set_map: Sequence, map_hash: int):
        self._set_arr ^= set_map

    def clone(self) -> NPGlobalCoverage:
        cpy = self.__class__(self._length)
        cpy.copy_from(self)
        return cpy

    def copy_from(self, other: NPGlobalCoverage):
        super().copy_from(other)
        np.copyto(self._set_arr, other._set_arr)

    def clear(self):
        self._set_arr.fill(0)

    def __eq__(self, other):
        return super().__eq__(other) and \
               np.array_equal(self._set_arr, other._set_arr)

    def __hash__(self):
        return hash(self._set_arr.data.tobytes())


HAMMING_LUT = np.zeros(256, dtype=np.uint8)
HAMMING_LUT[1] = 1
for i in range(1, 8):
    start = 1 << i
    for k in range(start):
        HAMMING_LUT[start + k] = 1 + HAMMING_LUT[k]

@cache
def _lookup():
    lookup = OrderedDict()
    lookup[0] = 0
    lookup[1] = 1 << 0
    lookup[2] = 1 << 1
    lookup[3] = 1 << 2
    lookup[4] = 1 << 3
    lookup[8] = 1 << 4
    lookup[16] = 1 << 5
    lookup[32] = 1 << 6
    lookup[128] = 1 << 7
    return lookup

@cache
def _count_class_lookup(count):
    res = 0
    for bn, lbl in _lookup().items():
        if count >= bn:
            res = lbl
        else:
            break
    return res

CLASS_LUT = np.zeros(256, dtype=np.uint8)
for i in range(len(CLASS_LUT)):
    CLASS_LUT[i] = _count_class_lookup(i)