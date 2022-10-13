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

    def __new__(cls, parent: CoverageState, set_map: Sequence, clr_map: Sequence, set_count: int, clr_count: int, map_hash: int, global_cov: GlobalCoverage):
        _hash = map_hash
        if cached := cls._cache.get(_hash):
            return cached
        new = super(CoverageState, cls).__new__(cls)
        new._parent = parent
        new._set_map = set_map
        new._set_count = set_count
        new._clr_map = clr_map
        new._clr_count = clr_count
        new._hash = _hash
        new._context = GlobalCoverage(global_cov._length)
        new._context.copy_from(global_cov)
        new._id = cls._id
        cls._id += 1
        # to obtain the context from the current global map, we revert the bits
        new._context.revert(set_map, clr_map)
        cls._cache[_hash] = new
        super(CoverageState, new).__init__()
        return new

    def __init__(self, parent: CoverageState, set_map: Sequence, clr_map: Sequence, set_count: int, clr_count: int, map_hash: int, global_cov: GlobalCoverage):
        pass

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        # return hash(self) == hash(other)
        return isinstance(other, CoverageState) and \
               hash(self) == hash(other)

    def __repr__(self):
        return f'({self._id}) +{self._set_count} -{self._clr_count}'
