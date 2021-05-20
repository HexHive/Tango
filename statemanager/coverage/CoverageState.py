from statemanager import StateBase
from input        import InputBase

class CoverageState(StateBase):
    def __init__(self):
        # TODO populate with coverage information
        pass

    def get_escaper(self) -> InputBase:
        # TODO generate a possible input to escape the current state
        # TODO (basically select an interesting input and mutate it)
        pass

    def update(self, sman: StateManager, transition: TransitionBase):
        # TODO update coverage and add interesting inputs
        pass

    def __hash__(self):
        pass

    def __eq__(self, other):
        pass