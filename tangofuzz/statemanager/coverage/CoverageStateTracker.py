from typing       import Callable
from loader       import StateLoaderBase
from statemanager import (StateBase,
                         StateTrackerBase,
                         CoverageState,
                         GlobalCoverage,
                         CoverageReader)
from input        import InputBase
from generator    import InputGeneratorBase
from uuid         import uuid4

class CoverageStateTracker(StateTrackerBase):
    def __init__(self, generator: InputGeneratorBase, loader: StateLoaderBase,
            bind_lib = None):
        super().__init__(generator)
        self._loader = loader
        self._bind_lib = bind_lib

        # session-unique shm file
        self._shm_name = f'/refuzz_cov_{uuid4()}'

        # set environment variables and load program with loader
        self._loader._exec_env.env.update({
            'REFUZZ_COVERAGE': self._shm_name
        })

    @classmethod
    async def create(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
        await self._loader.load_state(None, None)
        self._reader = CoverageReader(self._shm_name)

        # initialize a global coverage map
        self._global = GlobalCoverage(self._reader.length)

        self._current_state = None
        self.update(None, None)
        self._entry_state = self._current_state

        return self

    @property
    def entry_state(self) -> CoverageState:
        return self._entry_state

    @property
    def current_state(self) -> CoverageState:
        return self._current_state

    def update(self, input_gen: Callable[..., InputBase]):
        super().update(input_gen)

        prev = self._current_state

        coverage_map = self._reader.array
        set_list, clr_list = self._global.update(coverage_map)
        if set_list or clr_list:
            self._current_state = CoverageState(self._current_state, set_list, clr_list)

        new = self._current_state
        # we maintain _a_ path to each new state we encounter so that
        # reproducing the state is more path-aware and does not invoke a graph
        # search every time
        if prev != new and new.predecessor_transition is None:
            new.predecessor_transition = (prev, input_gen())

    def reset_state(self, state: StateBase):
        self._current_state = state
