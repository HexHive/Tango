from typing       import Callable
from loader       import StateLoaderBase
from statemanager import (StateBase,
                         StateTrackerBase,
                         CoverageState,
                         CoverageReader)
from input        import InputBase
from generator    import InputGeneratorBase

class CoverageStateTracker(StateTrackerBase):
    def __init__(self, generator: InputGeneratorBase, loader: StateLoaderBase):
        super().__init__(generator)

        # set environment variables and load program with loader
        loader._exec_env.env.update({
                'REFUZZ_COVERAGE': '/refuzz_cov'
            })
        loader.load_state(None, None)
        self._reader = CoverageReader('/refuzz_cov')
        self._entry_state = CoverageState(self._reader.array)

    @property
    def entry_state(self) -> CoverageState:
        return self._entry_state

    @property
    def current_state(self) -> CoverageState:
        return CoverageState(self._reader.array)

    def update_state(self, prev: StateBase, new: StateBase,
            input_gen: Callable[..., InputBase]):
        super().update_state(prev, new, input_gen)
        # TODO update other stuffz