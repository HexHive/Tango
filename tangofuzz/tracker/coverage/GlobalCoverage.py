from __future__   import annotations
from typing       import Sequence
from collections  import OrderedDict
from functools    import cache
from ctypes       import POINTER as P, c_ubyte as B, c_size_t as S, cast as C
import numpy as np

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

class GlobalCoverage:
    def __init__(self, length: int):
        self._length = length
        self._type = B * length
        self._set_arr = np.zeros(length, dtype=np.uint8)
        self.clear()

    def update(self, coverage_map: Sequence) -> (Sequence, int, int):
        kls_arr = CLASS_LUT[coverage_map]
        set_map = (self._set_arr | kls_arr) ^ self._set_arr
        self._set_arr ^= set_map
        set_count = np.sum(HAMMING_LUT[set_map])
        map_hash = hash(set_map.data.tobytes())

        return set_map, set_count, map_hash

    def revert(self, set_map: Sequence):
        self._set_arr ^= set_map

    def copy_from(self, other: GlobalCoverage):
        if self._length != other._length:
            raise RuntimeError("Mismatching coverage map sizes")
        np.copyto(self._set_arr, other._set_arr)

    def clear(self):
        self._set_arr.fill(0)

    def __eq__(self, other):
        return isinstance(other, GlobalCoverage) and \
               np.array_equal(self._set_arr, other._set_arr)

    def __hash__(self):
        return hash(self._set_arr.data.tobytes())
