from typing       import Callable
from loader       import StateLoaderBase
from statemanager import (StateBase,
                         StateTrackerBase,
                         CoverageState,
                         CoverageReader)
from input        import InputBase
from generator    import InputGeneratorBase

class CoverageStateTracker(StateTrackerBase):
    def __init__(self, generator: InputGeneratorBase, loader: StateLoaderBase,
            bind_lib = None):
        super().__init__(generator)

        self._bind_lib = bind_lib

        # set environment variables and load program with loader
        loader._exec_env.env.update({
                'REFUZZ_COVERAGE': '/refuzz_cov'
            })
        loader.load_state(None, None)
        self._reader = CoverageReader('/refuzz_cov')
        self._entry_state = CoverageState(self._reader.array,
            self._reader.address_of_buffer(self._reader._map),
            bind_lib=self._bind_lib)

    @property
    def entry_state(self) -> CoverageState:
        return self._entry_state

    @property
    def current_state(self) -> CoverageState:
        return CoverageState(self._reader.array,
            self._reader.address_of_buffer(self._reader._map),
            bind_lib=self._bind_lib)

    def update(self, prev: StateBase, new: StateBase,
            input_gen: Callable[..., InputBase]):
        super().update(prev, new, input_gen)
        # TODO update other stuffz
