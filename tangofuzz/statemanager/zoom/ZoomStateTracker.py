from typing       import Callable
from loader       import StateLoaderBase
from statemanager import (StateBase,
                         StateTrackerBase,
                         ZoomState,
                         ZoomStateReader)
from input        import InputBase
from generator    import InputGeneratorBase

class ZoomStateTracker(StateTrackerBase):
    def __init__(self, generator: InputGeneratorBase, loader: StateLoaderBase):
        super().__init__(generator)
        self._loader = loader

        # set environment variables and load program with loader
        self._loader._exec_env.env.update({
        })

    @classmethod
    async def create(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
        await self._loader.load_state(None, None)

        self._reader = ZoomStateReader()
        self._entry_state = ZoomState(self._reader.struct)
        return self

    @property
    def entry_state(self) -> ZoomState:
        return self._entry_state

    @property
    def current_state(self) -> ZoomState:
        state = ZoomState(self._reader.struct)
        state.state_manager = self.state_manager
        return state

    def update(self, prev: StateBase, new: StateBase,
            input_gen: Callable[..., InputBase]):
        super().update(prev, new, input_gen)
        # we maintain _a_ path to each new state we encounter so that
        # reproducing the state is more path-aware and does not invoke a graph
        # search every time
        if prev != new and new.predecessor_transition is None:
            new.predecessor_transition = (prev, input_gen())
        # TODO update other stuffz
