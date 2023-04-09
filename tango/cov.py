from __future__   import annotations
from . import debug, info, warning

from tango.core import (BaseState, BaseTracker, AbstractInput, LambdaProfiler,
    ValueMeanProfiler, CountProfiler, Path, BaseExplorer, BaseExplorerContext,
    LoadableTarget, BaseInput, get_profiler, TransmitInstruction,
    PreparedInput)
from tango.replay import ReplayLoader
from tango.unix import ProcessDriver, ProcessForkDriver, SharedMemoryObject
from tango.webui import WebRenderer, WebDataLoader
from tango.common import ComponentType, ComponentOwner
from tango.exceptions import LoadedException, StabilityException

from matplotlib.figure import Figure
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.colors import LogNorm
import seaborn as sns

from abc import ABC, abstractmethod
from typing       import Sequence, Callable, Optional
from collections  import OrderedDict
from functools    import cache, cached_property, partial
from uuid         import uuid4
import ctypes
from ctypes       import pythonapi, memset, memmove, sizeof, CDLL
from ctypes       import (POINTER as P,
                          c_bool as B,
                          c_ubyte as b,
                          c_uint32 as I,
                          c_size_t as S,
                          c_void_p as V,
                          c_char as c,
                          cast as C,
                          c_int,
                          byref, Structure)
import numpy as np
import io
import json
import asyncio
import sys
import os
import bisect

memcmp = pythonapi.memcmp
memcmp.argtypes = (V, V, S)
memcmp.restype = c_int

__all__ = [
    'FeatureSnapshot', 'FeatureMap', 'CoverageTracker',
    'CoverageDriver', 'CoverageForkDriver', 'CoverageWebRenderer',
    'CoverageWebDataLoader'
]

HAMMING_LUT = np.zeros(256, dtype=np.uint8)
HAMMING_LUT[1] = 1
for i in range(1, 8):
    start = 1 << i
    for k in range(start):
        HAMMING_LUT[start + k] = 1 + HAMMING_LUT[k]

@cache
def _lookup():
    lookup = OrderedDict()
    lookup[0] = 0
    lookup[1] = 1 << 0
    lookup[2] = 1 << 1
    lookup[3] = 1 << 2
    lookup[4] = 1 << 3
    lookup[8] = 1 << 4
    lookup[16] = 1 << 5
    lookup[32] = 1 << 6
    lookup[128] = 1 << 7
    return lookup

@cache
def _count_class_lookup(count):
    res = 0
    for bn, lbl in _lookup().items():
        if count >= bn:
            res = lbl
        else:
            break
    return res

CLASS_LUT = np.zeros(256, dtype=np.uint8)
for i in range(len(CLASS_LUT)):
    CLASS_LUT[i] = _count_class_lookup(i)

class FeatureSnapshot(BaseState):
    __slots__ = (
        '_parent', '_feature_mask', '_feature_count', '_feature_context',
        '_raw_coverage'
    )

    def __init__(self, parent: FeatureSnapshot,
            feature_mask: Sequence, feature_count: int, feature_map: FeatureMap,
            **kwargs):
        super().__init__(**kwargs)
        self._parent = parent
        self._feature_mask = feature_mask
        self._feature_count = feature_count
        self._feature_context = feature_map.clone()
        self._raw_coverage = feature_map._features.clone_object()

    def __eq__(self, other):
        # return hash(self) == hash(other)
        return isinstance(other, FeatureSnapshot) and \
               hash(self) == hash(other) and \
               self._feature_count == other._feature_count
               # np.array_equal(self._feature_mask, other._feature_mask) and \
               # self._feature_context == other._feature_context

    def __repr__(self):
        return f'({self._id}) +{self._feature_count}'

class FeatureMap(ABC):
    def __new__(cls, *args, **kwargs):
        if (bind_lib := kwargs.get('bind_lib')):
            return super(FeatureMap, CFeatureMap).__new__(CFeatureMap)
        else:
            return super(FeatureMap, NPFeatureMap).__new__(NPFeatureMap)

    def __init__(self, features: SharedMemoryObject, **kwargs):
        self._features = features
        self._shared_map = features.object

    @property
    def length(self):
        return self._features.ctype._length_

    @abstractmethod
    def extract(self, *, commit: bool) -> (Sequence, int, int):
        raise NotImplementedError

    @abstractmethod
    def commit(self, feature_mask: Sequence, mask_hash: int):
        raise NotImplementedError

    @abstractmethod
    def revert(self, feature_mask: Sequence, mask_hash: int):
        raise NotImplementedError

    @abstractmethod
    def reset(self, feature_mask: Sequence, mask_hash: int):
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> FeatureMap:
        raise NotImplementedError

    @abstractmethod
    def copy_from(self, other: FeatureMap):
        if self.length != other.length:
            raise RuntimeError("Mismatching coverage map sizes")

    @abstractmethod
    def clear(self):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        return isinstance(other, FeatureMap)

class CFeatureMap(FeatureMap):
    def __init__(self, *args, bind_lib, **kwargs):
        super().__init__(*args, **kwargs)
        self._feature_arr = self._features.ctype()
        self.clear()

        self._bind_lib = bind_lib
        self._bind_lib.diff.argtypes = (
            P(b), # global feature map (feature_arr)
            P(b), # local coverage array (coverage_map)
            P(b), # pre-allocated feature_mask output buffer
            S, # common size of the coverage buffers
            P(I), # pre-allocated feature_count output buffer
            P(I), # pre-allocated hash output buffer
            B, # boolean: apply in-place?
        )
        self._bind_lib.diff.restype = B # success?

        self._bind_lib.apply.argtypes = (
            P(b), # global feature map (feature_arr)
            P(b), # feature_mask buffer
            S # common size of the coverage buffers
        )
        self._bind_lib.apply.restype = B # success?

        self._bind_lib.reset.argtypes = (
            P(b), # global feature map (feature_arr)
            P(b), # feature_mask buffer
            S # common size of the coverage buffers
        )
        self._bind_lib.reset.restype = B # success?

    def extract(self, *, commit: bool=False) \
            -> (Sequence, int, int):
        feature_mask = self._features.ctype()
        feature_count = I()
        mask_hash = I()
        coverage_map = self._shared_map
        res = self._bind_lib.diff(self._feature_arr, coverage_map, feature_mask,
            self.length, byref(feature_count), byref(mask_hash), commit)
        return feature_mask, feature_count.value, mask_hash.value

    def commit(self, feature_mask: Sequence, mask_hash: int):
        self._bind_lib.apply(self._feature_arr, feature_mask, self.length)

    def revert(self, feature_mask: Sequence, mask_hash: int):
        self._bind_lib.apply(self._feature_arr, feature_mask, self.length)

    def reset(self, feature_mask: Sequence, mask_hash: int):
        self._bind_lib.reset(self._feature_arr, feature_mask, self.length)

    def clone(self) -> CFeatureMap:
        cpy = self.__class__(self._features, bind_lib=self._bind_lib)
        cpy.copy_from(self)
        return cpy

    def copy_from(self, other: CFeatureMap):
        super().copy_from(other)
        memmove(self._feature_arr, other._feature_arr, self.length)

    def clear(self):
        memset(self._feature_arr, 0, self.length)

    def __eq__(self, other):
        return super().__eq__(other) and \
            memcmp(self._feature_arr, other._feature_arr, self.length) == 0

class NPFeatureMap(FeatureMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._feature_arr = np.zeros(self.length, dtype=np.uint8)
        self.clear()

    def extract(self, *, commit: bool=False) \
            -> (Sequence, int, int):
        coverage_map = np.ctypeslib.as_array(self._shared_map)
        kls_arr = CLASS_LUT[coverage_map]
        feature_mask = (self._feature_arr | kls_arr) ^ self._feature_arr
        feature_count = np.sum(HAMMING_LUT[feature_mask])
        mask_hash = hash(feature_mask.data.tobytes())

        if commit:
            self._feature_arr ^= feature_mask
        return feature_mask, feature_count, mask_hash

    def commit(self, feature_mask: Sequence, mask_hash: int):
        self._feature_arr |= feature_mask

    def revert(self, feature_mask: Sequence, mask_hash: int):
        self._feature_arr ^= feature_mask

    def reset(self, feature_mask: Sequence, mask_hash: int):
        self._feature_arr &= feature_mask

    def clone(self) -> NPFeatureMap:
        cpy = self.__class__(self._features)
        cpy.copy_from(self)
        return cpy

    def copy_from(self, other: NPFeatureMap):
        super().copy_from(other)
        np.copyto(self._feature_arr, other._feature_arr)

    def clear(self):
        self._feature_arr.fill(0)

    def __eq__(self, other):
        return super().__eq__(other) and \
               np.array_equal(self._feature_arr, other._feature_arr)

class CoverageDriver(ProcessDriver,
        capture_paths=('driver.clear_coverage',)):
    @classmethod
    def match_config(cls, config: dict):
        return super().match_config(config) and \
            config['tracker'].get('type') == 'coverage'

    def __init__(self, *, clear_coverage: Optional[bool]=False, **kwargs):
        super().__init__(**kwargs)
        self._clear_cov = clear_coverage

    async def finalize(self, owner: ComponentOwner):
        # WARN this bypasses the expected component hierarchy and would usually
        # result in cyclic dependencies, but since both components are defined
        # and confined within this module, they are expected to be tightly
        # coupled and be aware of this dependency
        self._tracker: CoverageTracker = owner['tracker']
        assert isinstance(self._tracker, CoverageTracker)
        await super().finalize(owner)

    async def execute_input(self, input: AbstractInput):
        try:
            idx = 0
            async for instruction in input:
                idx += 1
                if self._clear_cov:
                    memset(self._tracker._features.object, 0,
                        self._tracker._features.size)
                await instruction.perform(self._channel)
                # we invalidate the current_state cache
                # WARN we use pop with a default value in case the cache is
                # invalidated twice in a row.
                self._tracker.__dict__.pop('current_state', None)
        except Exception as ex:
            raise LoadedException(ex, lambda: input[:idx]) from ex
        finally:
            ValueMeanProfiler("input_len", samples=100)(idx)
            CountProfiler("total_instructions")(idx)

# this class exists only to allow matching with forkserver==true
class CoverageForkDriver(CoverageDriver, ProcessForkDriver):
    pass

class CoverageTracker(BaseTracker,
        capture_components={ComponentType.driver},
        capture_paths=['tracker.native_lib', 'tracker.verify_raw_coverage',
            'tracker.track_heat', 'tracker.use_cmplog']):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['tracker'].get('type') == 'coverage'

    def __init__(self, *, driver: CoverageDriver, native_lib=None,
            verify_raw_coverage: bool=False, track_heat: bool=False,
            use_cmplog: bool=False, **kwargs):
        super().__init__(**kwargs)
        self._driver = driver
        self._verify = verify_raw_coverage
        self._track_heat = track_heat
        self._use_cmplog = use_cmplog

        if native_lib:
            self._bind_lib = CDLL(native_lib)
        elif (lib := os.getenv("TANGO_LIBDIR")):
            self._bind_lib = CDLL(os.path.join(lib, 'tango', 'pyfeaturemap.so'))
        else:
            self._bind_lib = None

        # session-unique shm file
        self._shm_uuid = os.getpid()

    async def finalize(self, owner: ComponentOwner):
        generator = owner['generator']
        startup = getattr(generator, 'startup_input', None)

        await self._driver.relaunch()
        if startup:
            await self._driver.execute_input(startup)

        self._features = SharedMemoryObject(
            f'/tango_cov_{self._shm_uuid}', lambda s: b * (s // sizeof(b)))
        info(f"Obtained coverage map {self._features._size=}")

        if self._use_cmplog:
            self._cmplog = CmpLogTables(self._shm_uuid)

        # initialize feature maps
        self._global = FeatureMap(self._features, bind_lib=self._bind_lib)
        self._scratch = FeatureMap(self._features, bind_lib=self._bind_lib)
        self._local = FeatureMap(self._features, bind_lib=self._bind_lib)
        if self._track_heat:
            self._differential = FeatureMap(self._features,
                                            bind_lib=self._bind_lib)
            self._diff_state = None
            self._feature_heat = np.zeros(self._differential.length, dtype=int)

        self._local_state = None
        self._current_state = None
        # the update creates a new initial _current_state
        self.update_state(None, input=None)
        self._entry_state = self._current_state

        await super().finalize(owner)

        # initialize local map and local state
        self.reset_state(self._current_state)

        LambdaProfiler('global_cov')(lambda: sum(
            map(lambda x: x._feature_count,
                filter(lambda x: x != self._entry_state,
                    self._state_graph.nodes
                )
            )
        ))

    @property
    def entry_state(self) -> FeatureSnapshot:
        return self._entry_state

    @cached_property
    def current_state(self) -> FeatureSnapshot:
        return self.peek(self._current_state, update_cache=False)

    def extract_snapshot(self, feature_map: FeatureMap,
            parent_state: FeatureSnapshot, *, commit: bool=True,
            allow_empty: bool=False, **kwargs) -> FeatureSnapshot:
        feature_mask, feature_count, mask_hash = feature_map.extract(commit=False)
        if feature_count or allow_empty:
            state = FeatureSnapshot(
                parent_state, feature_mask, feature_count, feature_map,
                tracker=self, state_hash=mask_hash, **kwargs)
            if commit:
                feature_map.commit(feature_mask, mask_hash)
            return state

    def update_state(self, source: FeatureSnapshot, /, *, input: AbstractInput,
            exc: Exception=None, peek_result: Optional[FeatureSnapshot]=None) \
            -> FeatureSnapshot:
        source = super().update_state(source, input=input, exc=exc,
                peek_result=peek_result)
        if not exc:
            if peek_result is None:
                next_state = self.extract_snapshot(self._global,
                    parent_state=source, allow_empty=source is None)
            else:
                # if peek_result was specified, we can skip the recalculation
                next_state = peek_result
                self._global.copy_from(next_state._feature_context)
                # we un-revert the bitmaps to obtain the actual global context
                self._global.revert(next_state._feature_mask, next_state._hash)

            if not next_state:
                next_state = source
            self._current_state = next_state

            # update local coverage
            self._update_local()

            # at the end, clear TORCs so that they remain attributable to
            #   the last input
            if self._use_cmplog:
                for torc in self._cmplog.torcs:
                    torc.object.LastIdx = torc.object.Length = 0

            return next_state
        else:
            if source:
                self._global.revert(source._feature_mask, source._hash)
            return source

    def peek(self,
            default_source: FeatureSnapshot=None,
            expected_destination: FeatureSnapshot=None,
            **kwargs) -> FeatureSnapshot:
        fmap = self._scratch
        if expected_destination:
            # when the destination is not None, we use its `context_map` as a
            # basis for calculating the coverage delta
            fmap.copy_from(expected_destination._feature_context)
            parent = expected_destination._parent
        else:
            fmap.copy_from(self._global)
            parent = default_source

        next_state = self.extract_snapshot(fmap, parent, commit=False,
            allow_empty=parent is None, **kwargs)
        if not next_state:
            next_state = FeatureSnapshot(parent, default_source._feature_mask,
                default_source._feature_count, default_source._feature_context,
                tracker=self, state_hash=hash(default_source), **kwargs)
        return next_state

    def reset_state(self, state: FeatureSnapshot):
        super().reset_state(state)
        self._current_state = state

        if self._verify:
            real_cov = np.asarray(self._features.object)
            state_cov = np.asarray(state._raw_coverage)
            if not np.array_equal(real_cov, state_cov):
                raise RuntimeError("State coverage did not match actual map.")

        # reset local maps
        self._local.clear()
        self.extract_snapshot(  # initialize the state of the local feature map
            self._local, None, update_cache=False)
        self._local_state = None
        if self._track_heat:
            self._differential.clear()
            self.extract_snapshot(
                self._differential, None, update_cache=False)
            self._diff_state = None
        # update the local maps with the latest coverage readings
        self._update_local()

        if self._use_cmplog:
            for torc in self._cmplog.torcs:
                torc.object.LastIdx = torc.object.Length = 0

    def _update_local(self):
        self._local_state = self.extract_snapshot(
            self._local, self._local_state,
            allow_empty=True, update_cache=False)
        if self._track_heat:
            self._diff_state = self.extract_snapshot(
                self._differential, self._diff_state,
                allow_empty=True, update_cache=False, commit=False)
            feature_mask = np.asarray(self._diff_state._feature_mask)
            hot_features, = feature_mask.nonzero()
            # FIXME check how often this is empty...
            self._feature_heat[hot_features] += 1

class CoverageReplayLoader(ReplayLoader,
        capture_components={'driver', 'tracker'},
        capture_paths=('loader.restore_cov_map',)):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['tracker'].get('type') == 'coverage'

    def __init__(self, *, driver: CoverageDriver, tracker: CoverageTracker,
            restore_cov_map: bool=True, **kwargs):
        super().__init__(driver=driver, tracker=tracker, **kwargs)
        self._restore = restore_cov_map

    async def apply_transition(self, transition: Transition,
            current_state: FeatureSnapshot, **kwargs) -> FeatureSnapshot:
        _, dst, _ = transition
        if self._restore and dst._parent:
            # inject the coverage map context of the expected dst
            self._tracker._features.write_object(dst._parent._raw_coverage)

        state = await super().apply_transition(transition, current_state,
            **kwargs)
        return state

class CoverageExplorer(BaseExplorer,
        capture_components={'tracker'},
        capture_paths=('explorer.exclude_uncolored', 'explorer.cmplog_samples',
                       'explorer.cmplog_goal')):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['tracker'].get('type') == 'coverage'

    def __init__(self, *, tracker: CoverageTracker,
            exclude_uncolored: bool=False,
            cmplog_samples: int=100, cmplog_goal: int=10, **kwargs):
        super().__init__(tracker=tracker, **kwargs)
        self._exclude_uncolored = exclude_uncolored
        self._cmplog_samples = cmplog_samples
        self._cmplog_goal = cmplog_goal

    def get_context_input(self, input: BaseInput, **kwargs) -> BaseExplorerContext:
        return CoverageExplorerContext(input, explorer=self, **kwargs)

    async def _minimize_transition(self, state_or_path: LoadableTarget,
            dst: FeatureSnapshot, input: BaseInput) -> BaseInput:
        try:
            inp = await super()._minimize_transition(state_or_path, dst, input)
        finally:
            # The state was initially found using a non-minimized path, which,
            # despite yielding the same feature set, could have a different
            # coverage map (features are bins). To force the state to be
            # updated, we delete it from the state cache.
            FeatureSnapshot.invalidate(dst)
        return inp

class CoverageExplorerContext(BaseExplorerContext):
    async def _handle_update(self, *args, **kwargs):
        if self._exp._tracker._use_cmplog:
            await self._perform_cmplog(*args, **kwargs)
        current_state, breadcrumbs = await super()._handle_update(
            *args, **kwargs)

    async def _perform_cmplog(self,
            updated, unseen,
            last_state, new_state, current_input):
        if not unseen:
            return

        torcs = [torc for torc in self._exp._tracker._cmplog.torcs
                if sizeof(torc.object.dtype) > 1]
        goal = self._exp._cmplog_goal
        threshold = self._exp._cmplog_samples
        entropy = self._exp._entropy
        while goal > 0 and threshold > 0:
            threshold -= 1
            while True:
                choose_torc, = entropy.choices(torcs,
                    weights=[sizeof(t.object.dtype) for t in torcs])
                length = max(choose_torc.object.LastIdx, choose_torc.object.Length)
                if length != 0:
                    break
            choose_idx = entropy.randint(0, length - 1)
            pair = choose_torc.object.Table[choose_idx]
            dtype = choose_torc.object.dtype

            # FIXME ensure these are copies in case dtype is not int
            arg1 = pair.Arg1
            arg2 = pair.Arg2
            try:
                rv = await self._identify_candidates(
                    last_state, new_state, current_input,
                    choose_torc, pair, dtype)

                if not rv:
                    continue

                pos1, pos2, colpos1, colpos2, colored = rv
                src, dst, inp = last_state, new_state, current_input
                subst = lambda x: \
                    inp[0:instr] + PreparedInput(instructions=(x,)) + inp[instr+1:]

                for pos, val in ((pos1, arg2), (pos2, arg1)):
                    for instr, off, size in pos:

                        orig = list(inp[instr])[0]
                        for variant in self._pattern_variants(val, dtype):
                            if len(variant) != size:
                                continue
                            data = bytearray(orig._data)
                            data[off:off + size] = variant
                            candidate = subst(TransmitInstruction(data)).flatten()
                            await self._exp.reload_state(src, dryrun=True)
                            try:
                                await self._exp._loader.apply_transition(
                                    (src, dst, candidate), src, update_cache=True)
                            except StabilityException as ex:
                                # we hit something other than dst, this is good!
                                interesting = ex.current_state not in \
                                    self._exp._tracker.state_graph
                                if interesting:
                                    goal -= 1
                                    self._exp._tracker.update_transition(
                                        last_state, ex.current_state, candidate,
                                        state_changed=True)
                                    info("Discovered new state with cmplog:"
                                        f" {ex.current_state}"
                                        f" {orig._data[off:off + size]} ->"
                                        f" {variant}")
                                    breadcrumbs = self._exp._current_path.copy()
                                    await self.update_transition(
                                        last_state, ex.current_state, candidate,
                                        orig_input=self.orig_input,
                                        breadcrumbs=breadcrumbs,
                                        state_changed=True, new_transition=True)
                            except Exception as ex:
                                # something horrible happened, ignore
                                continue
            except Exception as ex:
                continue

        await self._exp.reload_state(last_state, dryrun=True)
        await self._exp._loader.apply_transition(
            (last_state, new_state, current_input), last_state,
            update_cache=False)

    @staticmethod
    def intersectnd_nosort(a, b, /, *, axis=None):
        idx = np.indices((a.shape[axis or 0], b.shape[axis or 0]))
        if axis is not None:
            equals = lambda i, j: np.all(
                a.take(i, axis=axis) == b.take(j, axis=axis))
        else:
            equals = lambda i, j: a[i] == b[j]

        equals_ufn = np.frompyfunc(equals, 2, 1)
        match = equals_ufn(*idx)
        a_idx, b_idx = np.where(match)
        return a_idx, b_idx

    async def _identify_candidates(self, src, dst, inp, torc, pair, dtype):
        inp, orig_pos = self._matches_in_input(inp, pair, dtype)
        orig_pos1, orig_pos2 = orig_pos
        if not orig_pos1.size and not orig_pos2.size:
            return

        # colorize the input
        colored = inp
        orig_pos = list(orig_pos)
        for idx, pos in enumerate(orig_pos):
            instrs, split = np.unique(pos[:,0], return_index=True)
            offsets = np.split(pos[:,1:], split[1:])
            for i, instr in enumerate(instrs):
                colored, ranges = await self._colorize_input(src, dst,
                                                             colored, instr)
                if self._exp._exclude_uncolored:
                    to_delete = np.empty((0,), dtype=int)
                    for j, (off, size) in enumerate(offsets[i]):
                        if not any(off >= r.start and \
                                (off + size) <= r.stop for r in ranges):
                            # this match could not be colorized, so it's
                            # probably critical; discard it
                            to_delete = np.append(to_delete, split[i] + j)
                    orig_pos[idx] = np.delete(pos, to_delete, axis=0)

        # these might have been updated
        orig_pos1, orig_pos2 = orig_pos

        # find colored overlaps
        colored_pos1 = np.empty((0, 3), dtype=int)
        colored_pos2 = np.empty((0, 3), dtype=int)
        for i in range(max(torc.object.LastIdx, torc.object.Length)):
            cur_pair = torc.object.Table[i]
            _, (cpos1, cpos2) = self._matches_in_input(colored, cur_pair, dtype)
            colored_pos1 = np.vstack((colored_pos1, cpos1))
            colored_pos2 = np.vstack((colored_pos1, cpos2))

        if not colored_pos1.size and not colored_pos2.size:
            return

        debug(f"{colored_pos1.size=} and {colored_pos2.size=}")

        idx1, colidx1 = self.intersectnd_nosort(orig_pos1, colored_pos1, axis=0)
        idx2, colidx2 = self.intersectnd_nosort(orig_pos2, colored_pos2, axis=0)

        pos1 = np.unique(orig_pos1[idx1], axis=0)
        pos2 = np.unique(orig_pos2[idx2], axis=0)
        colpos1 = np.unique(colored_pos1[colidx1], axis=0)
        colpos2 = np.unique(colored_pos2[colidx2], axis=0)

        debug(f"{pos1.size=} and {pos2.size=}")

        return pos1, pos2, colpos1, colpos2, colored

    async def _colorize_input(self, src, dst, inp, instr):
        entropy = self._exp._entropy
        orig = list(inp[instr])[0]
        subst = lambda x: \
            inp[0:instr] + PreparedInput(instructions=(x,)) + inp[instr+1:]

        data = bytearray(orig._data)
        ranges = [slice(0, len(data))]
        colored_ranges = []
        while ranges:
            rng = ranges.pop()
            if rng.stop <= rng.start:
                break
            backup = data[rng]
            data[rng] = entropy.randbytes(rng.stop - rng.start)
            colored = subst(TransmitInstruction(data))
            await self._exp.reload_state(src, dryrun=True)
            try:
                await self._exp._loader.apply_transition(
                    (src, dst, colored), src, update_cache=False)
                inp = colored.flatten()
                bisect.insort(colored_ranges, rng,
                    key=lambda s: s.stop - s.start)
                need_reapply = False
            except Exception:
                data[rng] = backup
                bisect.insort(ranges, slice(rng.start, rng.stop // 2),
                    key=lambda s: s.stop - s.start)
                bisect.insort(ranges, slice(rng.stop // 2 + 1, rng.stop),
                    key=lambda s: s.stop - s.start)
                need_reapply = True

        if need_reapply:
            await self._exp.reload_state(src, dryrun=True)
            await self._exp._loader.apply_transition((src, dst, inp), src,
                update_cache=False)
        return inp, colored_ranges

    @classmethod
    def _matches_in_input(cls, inp, pair, dtype):
        vars1 = cls._pattern_variants(pair.Arg1, dtype)
        vars2 = cls._pattern_variants(pair.Arg2, dtype)

        pos1 = np.empty((0, 3), dtype=int)
        pos2 = np.empty((0, 3), dtype=int)

        # first we search in individual instructions
        for idx, instr in enumerate(inp):
            if not isinstance(instr, TransmitInstruction):
                continue
            for variant in vars1:
                arrs = tuple(np.array(((idx, offset, len(variant)),), dtype=int)
                    for offset in cls._find_pattern(variant, instr._data))
                if arrs:
                    pos1 = np.vstack((pos1, np.concatenate(arrs)))
            for variant in vars2:
                arrs = tuple(np.array(((idx, offset, len(variant)),), dtype=int)
                    for offset in cls._find_pattern(variant, instr._data))
                if arrs:
                    pos2 = np.vstack((pos2, np.concatenate(arrs)))

        # otherwise, value might be across boundaries
        if not pos1.size and not pos2.size:
            data = b''.join(x._data
                for x in inp if isinstance(x, TransmitInstruction))
            inp = PreparedInput(instructions=(TransmitInstruction(data),))
            for variant in vars1:
                arrs = tuple(np.array(((idx, offset, len(variant)),), dtype=int)
                    for offset in cls._find_pattern(variant, data))
                if arrs:
                    pos1 = np.vstack((pos1, np.concatenate(arrs)))
            for variant in vars2:
                arrs = tuple(np.array(((idx, offset, len(variant)),), dtype=int)
                    for offset in cls._find_pattern(variant, data))
                if arrs:
                    pos2 = np.vstack((pos2, np.concatenate(arrs)))

        return inp, (pos1, pos2)

    @classmethod
    def _pattern_variants(cls, val, dtype):
        if dtype in (ctypes.c_uint8, ctypes.c_uint16,
                     ctypes.c_uint32, ctypes.c_uint64):
            yield from cls._int_pattern_variants(val, dtype)

    @classmethod
    def _int_pattern_variants(cls, val, dtype):
        nbytes = sizeof(dtype)
        nbits = nbytes * 8
        msb_mask = 1 << (nbits - 1)
        bo_variants = ('little',)
        if val != 0:
            bo_variants = ('little', 'big')
        for byteorder in bo_variants:
            signed_variants = [val]
            if val & msb_mask:
                signed_variants.append(-val)
            for sval in signed_variants:
                for rval in (sval-1, sval, sval+1):
                    for n in range(nbytes):
                        try:
                            yield rval.to_bytes(n + 1, byteorder=byteorder,
                                                signed=rval < 0)
                        except OverflowError:
                            continue
                    # check for string literal ints
                    yield str(rval).encode()

    @classmethod
    def _find_pattern(cls, pattern, data):
        start = 0
        while True:
            offset = data.find(pattern, start)
            if offset == -1:
                break
            yield offset
            start = offset + 1

class CoverageWebRenderer(WebRenderer,
        capture_components=('tracker',)):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['tracker'].get('type') == 'coverage'

    def __init__(self, *args, tracker: CoverageTracker, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracker = tracker

    def get_webui_factory(self):
        return partial(CoverageWebDataLoader, tracker=self._tracker,
                       **self._webui_kwargs)

class CoverageWebDataLoader(WebDataLoader):
    def __init__(self, *args, tracker: CoverageTracker,
            draw_graph: bool=True, draw_heatmap: bool=False,
            stats_update_period: float=1, **kwargs):
        self._tracker = tracker
        tmp = []
        if draw_heatmap and tracker._track_heat and stats_update_period >= 0:
            draw_graph = False
            tmp.append(asyncio.create_task(
                    get_profiler('perform_instruction').listener(
                        period=stats_update_period)(self.update_heatmap)))

        super().__init__(*args, draw_graph=draw_graph,
                         stats_update_period=stats_update_period, **kwargs)

        self.tasks.extend(tmp)

    def draw_heatmap(self, array):
        # Filter out low-frequency regions
        threshold = 0.01 * np.max(array)
        filtered_array = array[array >= threshold]

        # Desired fixed-size 2-D grid shape
        grid_shape = (10, 10)

        # Calculate the number of elements to group for averaging
        group_size = int(np.ceil(len(filtered_array) / (grid_shape[0] * grid_shape[1])))

        # Reshape the 1-D array into groups
        padded_array = np.pad(filtered_array, (0, group_size * grid_shape[0] * grid_shape[1] - len(filtered_array)), mode='constant')
        grouped_array = np.reshape(padded_array, (-1, group_size))

        # Calculate the average of grouped elements
        averaged_array = np.mean(grouped_array, axis=1)

        # Reshape the averaged array into the fixed-size 2-D grid
        rescaled_array = np.reshape(averaged_array, grid_shape)

        # Set minimum value to a small positive number to avoid issues with log scale
        min_value = np.min(rescaled_array[rescaled_array > 0], initial=1)
        rescaled_array[rescaled_array == 0] = min_value

        # Plot the heatmap
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(rescaled_array, cmap='viridis', interpolation='bicubic', aspect='auto', norm=LogNorm())
        ax.set_title('Code Region Execution Frequency Heatmap')
        fig.colorbar(im, ax=ax, label='Execution Frequency')

        # Create a buffer to store the SVG data
        svg_buffer = io.StringIO()

        # Create a canvas object and save the figure to the buffer
        canvas = FigureCanvasSVG(fig)
        canvas.print_figure(svg_buffer)

        # Get the SVG string from the buffer
        svg_string = svg_buffer.getvalue()

        # Close the buffer
        svg_buffer.close()
        return svg_string

    async def update_heatmap(self, *args, ret=None, **kwargs):
        array = self._tracker._feature_heat
        if not array.size:
            return
        svg = self.draw_heatmap(array)
        msg = json.dumps({
            'cmd': 'update_painting',
            'items': {
                'svg': svg
            }
        })
        await self._ws.send_str(msg)

class CmpLogTables:
    def __init__(self, uuid):
        self.TORC1 = SharedMemoryObject(
            f'/tango_torc1_{uuid}',
            lambda s: self.get_table_type(ctypes.c_uint8))
        self.TORC2 = SharedMemoryObject(
            f'/tango_torc2_{uuid}',
            lambda s: self.get_table_type(ctypes.c_uint16))
        self.TORC4 = SharedMemoryObject(
            f'/tango_torc4_{uuid}',
            lambda s: self.get_table_type(ctypes.c_uint32))
        self.TORC8 = SharedMemoryObject(
            f'/tango_torc8_{uuid}',
            lambda s: self.get_table_type(ctypes.c_uint64))

    @property
    def torcs(self):
        yield from (getattr(self, name)
            for name in ('TORC1', 'TORC2', 'TORC4', 'TORC8'))

    @staticmethod
    def get_table_type(dtyp, capacity=1024):
        class Pair(Structure):
            _fields_ = (
                ('Arg1', dtyp),
                ('Arg2', dtyp),
            )
        class TableOfRecentCompares(Structure):
            _fields_ = (
                ('Length', S),
                ('LastIdx', S),
                ('Table', Pair * capacity),
            )
            dtype = dtyp
        return TableOfRecentCompares
