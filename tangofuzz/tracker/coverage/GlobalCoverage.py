from statemanager import StateBase, StateManager
from input        import InputBase
from typing       import Sequence
from collections  import OrderedDict
from functools    import cache, reduce, partial
from ctypes       import POINTER as P, c_ubyte as B, c_size_t as S, cast as C

class GlobalCoverage:
    def __init__(self, length: int):
        self._set_arr = (B * length)()
        self._clr_arr = (B * length)(*((0xFF,) * length))

    def update(self, coverage_map: Sequence) -> (set, set):
        # update the global coverage maps
        set_list = set()
        clr_list = set()
        for i, c in enumerate(coverage_map):
            kls = self._count_class_lookup(c)
            if self._set_arr[i] != (self._set_arr[i] := self._set_arr[i] | kls):
                set_list.add((i, kls))
            if self._clr_arr[i] != (self._clr_arr[i] := self._clr_arr[i] & kls):
                clr_list.add((i, kls))
        return set_list, clr_list

    @staticmethod
    @cache
    def lookup():
        lookup = OrderedDict()
        lookup[0] = 0
        lookup[1] = 1
        lookup[2] = 2
        lookup[3] = 4
        lookup[4] = 8
        lookup[8] = 16
        lookup[16] = 32
        lookup[32] = 64
        lookup[128] = 128
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