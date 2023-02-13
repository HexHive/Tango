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
        self._global = GlobalCoverage(self._reader.length, bind_lib=self._bind_lib)
        self._scratch = GlobalCoverage(self._reader.length, bind_lib=self._bind_lib)
        self._local = GlobalCoverage(self._reader.length, bind_lib=self._bind_lib)

        self._local_state = None
        self._current_state = None
        # the update creates a new initial _current_state
        self.update(None, None)
        self._entry_state = self._current_state

        # initialize local map and local state
        self.reset_state(self._current_state)

        return self

    @property
    def entry_state(self) -> CoverageState:
        return self._entry_state

    @property
    def current_state(self) -> CoverageState:
        return self.peek(self._current_state)

    def _diff_global_to_state(self, global_map: GlobalCoverage, parent_state: StateBase,
            local_state: bool=False, allow_empty: bool=False) -> StateBase:
        coverage_map = self._reader.array
        set_map, set_count, map_hash = global_map.update(coverage_map)
        if set_count or allow_empty:
            return CoverageState(parent_state, set_map, set_count, map_hash, global_map, do_not_cache=local_state)
        else:
            return None

    def update(self, source: StateBase, input: InputBase, peek_result: StateBase=None) -> StateBase:
        if peek_result is None:
            next_state = self._diff_global_to_state(self._global, parent_state=source,
                allow_empty=source is None)
        else:
            # if peek_result was specified, we can skip the recalculation
            next_state = peek_result
            self._global.copy_from(next_state._context)
            # we un-revert the bitmaps to obtain the actual global context
            self._global.revert(next_state._set_map, next_state._hash)

        if not next_state or (same := next_state == source):
            next_state = source
            same = True
        else:
            self._current_state = next_state

        # we maintain _a_ path to each new state we encounter so that
        # reproducing the state is more path-aware and does not invoke a graph
        # search every time
        if input is not None and not same and next_state.predecessor_transition is None:
            next_state.predecessor_transition = (source, input)

        # update local coverage
        self._update_local()

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

        next_state = self._diff_global_to_state(glbl, parent, allow_empty=parent is None)
        if not next_state:
            next_state = default_source
        return next_state

    def reset_state(self, state: StateBase):
        self._current_state = state

        # reset local map
        self._local.clear()
        self._local_state = None
        # update the local map with the latest coverage readings
        self._update_local()

    def _update_local(self):
        # this is a pseudo-state that stores the last observed diffs in the local map
        next_state = self._diff_global_to_state(self._local, self._local_state,
            local_state=True, allow_empty=True)
        # TODO is there special handling needed if next_state is None?
        self._local_state = next_state
