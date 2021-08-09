from statemanager import StateBase, StateManager
from input        import InputBase
from typing       import Sequence
from collections  import OrderedDict
from functools    import cache, reduce

class CoverageState(StateBase):
    def __init__(self, coverage_map: Sequence):
        super().__init__()
        # populate with AFL-style global coverage information
        self._cov = {
            edge: self._count_class_lookup(count)
            for edge, count in enumerate(coverage_map)
            if count != 0
        }

    @staticmethod
    @cache
    def lookup():
        lookup = OrderedDict.fromkeys([0, 1, 2, 3, 4, 8, 16, 32, 128])
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

    def __hash__(self):
        return reduce(lambda x,y: x ^ hash(y), self._cov.items(), 0)

    def __eq__(self, other):
        return self._cov == other._cov

    def __repr__(self):
        return f'CoverageState({repr(self._cov)})'
