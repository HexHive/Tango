from __future__   import annotations
from typing       import Sequence
from collections  import OrderedDict
from functools    import cache
from ctypes       import POINTER as P, c_ubyte as B, c_size_t as S, cast as C
from ctypes       import memmove, memset, sizeof

class GlobalCoverage:
    def __init__(self, length: int):
        self._length = length
        self._set_arr = (B * length)()
        self._clr_arr = (B * length)()
        self.clear()

    def update(self, coverage_map: Sequence) -> (set, set):
        # update the global coverage maps
        set_list = set()
        clr_list = set()
        for i, c in enumerate(coverage_map):
            kls = self._count_class_lookup(c)

            prev_set = self._set_arr[i]
            self._set_arr[i] = self._set_arr[i] | kls
            xor = self._set_arr[i] ^ prev_set
            if xor:
                set_list.add((i, kls))

            # prev_clr = self._clr_arr[i]
            # self._clr_arr[i] = self._clr_arr[i] & kls
            # xor = self._clr_arr[i] ^ prev_clr
            # for j in range(sizeof(B) * 8):
            #     if xor & 1:
            #         clr_list.add((i, 1 << j))
            #     xor >>= 1

        return frozenset(set_list), frozenset(clr_list)

    @staticmethod
    @cache
    def lookup():
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

    @classmethod
    @cache
    def _count_class_lookup(cls, count):
        res = 0
        for bn, lbl in cls.lookup().items():
            if count >= bn:
                res = lbl
            else:
                break
        return res

    def copy_from(self, other: GlobalCoverage):
        if self._length != other._length:
            raise RuntimeError("Mismatching coverage map sizes")
        memmove(self._set_arr, other._set_arr, self._length)
        memmove(self._clr_arr, other._clr_arr, self._length)

    def clear(self):
        memset(self._set_arr, 0, self._length)
        memset(self._clr_arr, 0xFF, self._length)