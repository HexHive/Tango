from typing       import Callable
from loader       import StateLoaderBase
from statemanager import (StateBase,
                         StateTrackerBase,
                         StackState,
                         StackHashReader)
from input        import InputBase
from generator    import InputGeneratorBase
from uuid         import uuid4

class StackStateTracker(StateTrackerBase):
    def __init__(self, generator: InputGeneratorBase, loader: StateLoaderBase):
        super().__init__(generator)

        # session-unique shm file
        cov_shm_name = f'/refuzz_cov_{uuid4()}'
        stk_shm_name = f'/refuzz_stack_{uuid4()}'

        # set environment variables and load program with loader
        loader._exec_env.env.update({
                'REFUZZ_COVERAGE': cov_shm_name,
                'REFUZZ_STACK': stk_shm_name
            })
        loader.load_state(None, None)
        self._reader = StackHashReader(stk_shm_name)
        self._entry_state = StackState(int(self._reader.array.value or 0))

    @property
    def entry_state(self) -> StackState:
        return self._entry_state

    @property
    def current_state(self) -> StackState:
        return StackState(int(self._reader.array.value or 0))

    def update(self, prev: StateBase, new: StateBase,
            input_gen: Callable[..., InputBase]):
        super().update(prev, new, input_gen)
        # we maintain _a_ path to each new state we encounter so that
        # reproducing the state is more path-aware and does not invoke a graph
        # search every time
        if prev != new and new.predecessor_transition is None:
            new.predecessor_transition = (prev, input_gen())
        # TODO update other stuffz
