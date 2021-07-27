from loader       import StateLoaderBase
from statemanager import (StateTrackerBase,
                         CoverageState,
                         CoverageReader,
                         PreparedTransition)
from input        import InputBase, PreparedInput

class CoverageStateTracker(StateTrackerBase):
    def __init__(self, loader: StateLoaderBase):
        # set environment variables and load program with loader
        loader._exec_env.env.update({
                'REFUZZ_COVERAGE': '/refuzz_cov'
            })
        loader.load_state(None, None)
        self._reader = CoverageReader('/refuzz_cov')
        self._state0 = CoverageState(self._reader.array)

    @property
    def initial_state(self) -> CoverageState:
        return self._state0

    @property
    def initial_transition(self) -> PreparedTransition:
        return PreparedTransition(PreparedInput())

    @property
    def current_state(self) -> CoverageState:
        # TODO access shmem region or IPC to get coverage info
        return CoverageState(self._reader.array)
        pass

    def update_state(self, state: CoverageState, input: InputBase):
        pass