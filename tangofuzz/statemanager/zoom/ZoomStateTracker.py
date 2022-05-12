from typing       import Callable
from loader       import StateLoaderBase
from statemanager import (StateBase,
                         StateTrackerBase,
                         ZoomState)
from input        import InputBase
from generator    import InputGeneratorBase

class ZoomStateTracker(StateTrackerBase):
    def __init__(self, generator: InputGeneratorBase, loader: StateLoaderBase):
        super().__init__(generator)

        # set environment variables and load program with loader
        loader._exec_env.env.update({
            })
        loader.load_state(None, None)
        self._entry_state = ZoomState()

    @property
    def entry_state(self) -> ZoomState:
        return self._entry_state

    @property
    def current_state(self) -> ZoomState:
        return ZoomState()

    def update(self, prev: StateBase, new: StateBase,
            input_gen: Callable[..., InputBase]):
        super().update(prev, new, input_gen)
        # we maintain _a_ path to each new state we encounter so that
        # reproducing the state is more path-aware and does not invoke a graph
        # search every time
        if prev != new and new.predecessor_transition is None:
            new.predecessor_transition = (prev, input_gen())
        # TODO update other stuffz
