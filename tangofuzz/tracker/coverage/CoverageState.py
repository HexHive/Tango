from __future__   import annotations
from tracker import StateBase
from tracker.coverage import GlobalCoverage
from input        import InputBase
from typing       import Sequence
from collections  import OrderedDict
from functools    import cache, reduce, partial
from ctypes       import POINTER as P, c_ubyte as B, c_size_t as S, cast as C

class CoverageState(StateBase):
    _cache = {}
    _id = 0

    def __new__(cls, parent: CoverageState, set_list: frozenset, clr_list: frozenset, global_cov: GlobalCoverage):
        _hash = hash((set_list, clr_list))
        if cached := cls._cache.get(_hash):
            return cached
        new = super(CoverageState, cls).__new__(cls)
        new._parent = parent
        new._set_list = set_list
        new._clr_list = clr_list
        new._hash = _hash
        new._context = GlobalCoverage(global_cov._length)
        new._context.copy_from(global_cov)
        new._id = cls._id
        cls._id += 1
        # to obtain the context from the current global map, we revert the bits
        for s in set_list:
            new._context._set_arr[s[0]] &= ~s[1]
        for c in clr_list:
            new._context._clr_arr[c[0]] |= c[1]
        cls._cache[_hash] = new
        super(CoverageState, new).__init__()
        return new

    def __init__(self, parent: CoverageState, set_list: frozenset, clr_list: frozenset, global_cov: GlobalCoverage):
        pass

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        # return hash(self) == hash(other)
        return isinstance(other, CoverageState) and \
               self._set_list == other._set_list and \
               self._clr_list == other._clr_list

    def __repr__(self):
        return f'({self._id}) +{len(self._set_list)} -{len(self._clr_list)}'
