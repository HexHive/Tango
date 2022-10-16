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
        self._shm_uuid = uuid4()
        self._shm_name = f'/refuzz_cov_{self._shm_uuid}'
        self._shm_size_name = f'/refuzz_size_{self._shm_uuid}'

        # set environment variables and load program with loader
        self._loader._exec_env.env.update({
            'REFUZZ_COVERAGE': self._shm_name,
            'REFUZZ_SIZE': self._shm_size_name
        })

    @classmethod
    async def create(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
        await self._loader.load_state(None, None)
        self._reader = CoverageReader(self._shm_name, self._shm_size_name)

        # initialize a global coverage map
        self._global = GlobalCoverage(self._reader.length)
        self._scratch = GlobalCoverage(self._reader.length)

        self._current_state = None
        self.update(None, None)
        self._entry_state = self._current_state

        return self

    @property
    def entry_state(self) -> CoverageState:
        return self._entry_state

    @property
    def current_state(self) -> CoverageState:
        return self.peek(self._current_state)

    def _diff_global_to_state(self, global_map: GlobalCoverage, parent_state: StateBase) -> StateBase:
        coverage_map = self._reader.array
        set_map, clr_map, set_count, clr_count, map_hash = global_map.update(coverage_map)
        if set_count or clr_count:
            return CoverageState(parent_state, set_map, clr_map, set_count, clr_count, map_hash, global_map)
        else:
            return None

    def update(self, source: StateBase, input: InputBase) -> StateBase:
        parent = source
        glbl = self._global

        next_state = self._diff_global_to_state(glbl, parent)
        if not next_state:
            next_state = source
        else:
            self._current_state = next_state

        # we maintain _a_ path to each new state we encounter so that
        # reproducing the state is more path-aware and does not invoke a graph
        # search every time
        if input is not None and source != next_state and next_state.predecessor_transition is None:
            next_state.predecessor_transition = (source, input)

        return next_state

    def peek(self, default_source: StateBase=None, expected_destination: StateBase=None) -> StateBase:
        glbl = self._scratch
        if expected_destination:
            # when the destination is not None, we use its `context_map` as a
            # basis for calculating the coverage delta
            glbl.copy_from(expected_destination._context)
            parent = expected_destination._parent
        else:
            glbl.copy_from(self._global)
            parent = default_source

        next_state = self._diff_global_to_state(glbl, parent)
        if not next_state:
            next_state = default_source
        return next_state

    def reset_state(self, state: StateBase):
        self._current_state = state
