from typing       import Callable
from tracker import StateBase, LoaderDependentTracker
from tracker.coverage import CoverageState, GlobalCoverage, CoverageReader
from input        import InputBase
from uuid         import uuid4

class CoverageStateTracker(LoaderDependentTracker):
    def __init__(self, /, *, bind_lib=None, **kwargs):
        super().__init__(**kwargs)
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
        self._scratch = GlobalCoverage(self._reader.length)

        self._current_state = None
        self.update(None, None, None)
        self._entry_state = self._current_state

        return self

    @property
    def entry_state(self) -> CoverageState:
        return self._entry_state

    @property
    def current_state(self) -> CoverageState:
        return self._current_state

    def update(self, source: StateBase, destination: StateBase, input_gen: Callable[..., InputBase], dryrun: bool=False) -> StateBase:
        # assuming the destination is not None, we use its `context_map` as a
        # basis for calculating the coverage delta
        parent = source
        if destination:
            assert dryrun, "destination is not None while dryrun == False"
            glbl = self._scratch
            glbl.copy_from(destination._context)
            parent = destination._parent
        elif dryrun:
            glbl = self._scratch
            glbl.copy_from(self._global)
        else:
            glbl = self._global

        coverage_map = self._reader.array
        set_list, clr_list = glbl.update(coverage_map)
        new = source
        if set_list or clr_list:
            new = CoverageState(parent, set_list, clr_list, glbl)
            if not dryrun:
                self._current_state = new

        # we maintain _a_ path to each new state we encounter so that
        # reproducing the state is more path-aware and does not invoke a graph
        # search every time
        if not dryrun and input_gen and source != new and new.predecessor_transition is None:
            new.predecessor_transition = (source, input_gen())

        return new

    def reset_state(self, state: StateBase):
        self._current_state = state
