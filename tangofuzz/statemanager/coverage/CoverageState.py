from statemanager import StateBase, StateManager
from input        import InputBase
from typing       import Sequence
from collections  import OrderedDict
from functools    import cache, reduce, partial
from ctypes       import POINTER as P, c_ubyte as B, c_size_t as S, cast as C

class CoverageState(StateBase):
    _cache = {}

    def __new__(cls, coverage_map: Sequence, address: int, bind_lib):
        if not bind_lib:
            # populate with AFL-style global coverage information
            cov = {
                edge: cls._count_class_lookup(count)
                for edge, count in enumerate(coverage_map)
                if count != 0
            }
            _hash = cls._hash_cov(cov, address)
            if cached := cls._cache.get(_hash):
                return cached
            new = super(CoverageState, cls).__new__(cls)
            new._cov = cov
            new._map = address
            cls._cache[_hash] = new
            super(CoverageState, new).__init__()
            return new
        else:
            _hash = bind_lib.hash_cov(C(address, P(B)), S(len(coverage_map)))
            if cached := cls._cache.get(_hash):
                return cached
            new = super(CoverageState, cls).__new__(cls)
            new._cov = coverage_map
            new._map = address
            new._hash_cov = lambda cov, map: _hash
            cls._cache[_hash] = new
            super(CoverageState, new).__init__()
            return new

    def __init__(self, coverage_map: Sequence, address: int, bind_lib):
        pass

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


    def get_escaper(self) -> InputBase:
        # TODO generate a possible input to escape the current state
        # TODO (basically select an interesting input and mutate it)
        pass

    def update(self, sman: StateManager, input: InputBase):
        # TODO update coverage and add interesting inputs
        pass

    @classmethod
    def _hash_cov(cls, cov, map):
        return reduce(lambda x,y: x ^ hash(y), cov.items(), 0)

    def __hash__(self):
        return self._hash_cov(self._cov, self._map)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return f'CoverageState({hex(hash(self))})'
