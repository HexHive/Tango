from loader       import StateLoaderBase
from statemanager import StateTrackerBase,
                         CoverageState

class CoverageStateTracker(StateTrackerBase):
    def __init__(self, loader: StateLoaderBase):
        # TODO set environment variables and load program with loader
        pass

    @property
    def initial_state(self) -> CoverageState:
        pass

    @property
    def current_state(self) -> CoverageState:
        # TODO access shmem region or IPC to get coverage info
        pass