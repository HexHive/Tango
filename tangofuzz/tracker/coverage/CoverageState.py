from __future__   import annotations
from statemanager import StateBase, StateManager
from input        import InputBase
from typing       import Sequence
from collections  import OrderedDict
from functools    import cache, reduce, partial
from ctypes       import POINTER as P, c_ubyte as B, c_size_t as S, cast as C

class CoverageState(StateBase):
    _cache = {}

    def __new__(cls, parent: CoverageState, set_list: set, clr_list: set):
        _hash = hash((parent, set_list, clr_list))
        if cached := cls._cache.get(_hash):
            return cached
        new = super(CoverageState, cls).__new__(cls)
        new._parent = parent
        new._set_list = set_list
        new._clr_list = clr_list
        new._hash = _hash
        cls._cache[_hash] = new
        super(CoverageState, new).__init__()
        return new

    def __init__(self, parent: CoverageState, set_list: set, clr_list: set):
        pass

    def get_escaper(self) -> InputBase:
        # TODO generate a possible input to escape the current state
        # TODO (basically select an interesting input and mutate it)
        pass

    def update(self, sman: StateManager, input: InputBase):
        # TODO update coverage and add interesting inputs
        pass

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        # return hash(self) == hash(other)
        return self._parent == other._parent and \
               self._set_list == other._set_list and \
               self._clr_list == other._clr_list

    def __repr__(self):
        return f'CoverageState({hex(hash(self))})'
