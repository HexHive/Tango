from loader       import StateLoaderBase
from statemanager import (StateTrackerBase,
                         CoverageState,
                         PreparedTransition)
from input        import InputBase, PreparedInput

class CoverageStateTracker(StateTrackerBase):
    def __init__(self, loader: StateLoaderBase):
        # TODO set environment variables and load program with loader
        pass

    @property
    def initial_state(self) -> CoverageState:
        pass

    @property
    def initial_transition(self) -> PreparedTransition:
        return PreparedTransition(PreparedInput())

    @property
    def current_state(self) -> CoverageState:
        # TODO access shmem region or IPC to get coverage info
        pass

    def update_state(self, state: CoverageState, input: InputBase):
        pass