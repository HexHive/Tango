from statemanager import StateBase, StateManager
from input        import InputBase

class ZoomState(StateBase):
    def __init__(self):
        super().__init__()

    def get_escaper(self) -> InputBase:
        # TODO generate a possible input to escape the current state
        # TODO (basically select an interesting input and mutate it)
        pass

    def update(self, sman: StateManager, input: InputBase):
        # TODO update coverage and add interesting inputs
        pass

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return f'ZoomState({hex(hash(self))})'
