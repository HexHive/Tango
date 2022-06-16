from statemanager import StateBase, StateManager, ZoomFeedback
from input        import InputBase
from ctypes import pointer

class ZoomState(StateBase):
    _cache = {}

    CELL_DIMENSION = 50

    def __new__(cls, struct: ZoomFeedback):
        _hash = cls._calc_hash(struct)
        if cached := cls._cache.get(_hash):
            return cached
        new = super(ZoomState, cls).__new__(cls)
        super(ZoomState, new).__init__()
        new._struct = cls._copy_struct(struct)
        new._hash = _hash
        cls._cache[_hash] = new
        return new

    def __init__(self, struct: ZoomFeedback):
        pass

    @classmethod
    def _copy_struct(cls, struct):
        dst = type(struct)()
        pointer(dst)[0] = struct
        return dst

    def get_escaper(self) -> InputBase:
        # TODO generate a possible input to escape the current state
        # TODO (basically select an interesting input and mutate it)
        pass

    def update(self, sman: StateManager, input: InputBase):
        # TODO update coverage and add interesting inputs
        pass

    @classmethod
    def _calc_hash(cls, struct: ZoomFeedback):
        return hash((struct.player_location.x // cls.CELL_DIMENSION,
                     struct.player_location.y // cls.CELL_DIMENSION))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        # return f'ZoomState({hex(hash(self))})'
        return (f'x={int(self._struct.player_location.x // self.CELL_DIMENSION)}\n'
                f'y={int(self._struct.player_location.y // self.CELL_DIMENSION)}\n'
                f'z={int(self._struct.player_location.z // 8)}')