from statemanager import StateBase, StateManager
from input        import InputBase

class StackState(StateBase):
    _cache = {}

    def __new__(cls, stack_hash):
        if cached := cls._cache.get(stack_hash):
            return cached
        new = super(StackState, cls).__new__(cls)
        new._hash = stack_hash
        cls._cache[stack_hash] = new
        super(StackState, new).__init__()
        return new

    def __init__(self, stack_hash):
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
        return hash(self) == hash(other)

    def __repr__(self):
        return f'StackState({hex(hash(self))})'
