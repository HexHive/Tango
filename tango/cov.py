from __future__   import annotations
from . import debug, info, warning

from tango.core import (BaseState, BaseTracker, AbstractInput, LambdaProfiler,
    ValueMeanProfiler, CountProfiler, Path, BaseExplorer, LoadableTarget,
    BaseInput, get_profiler)
from tango.replay import ReplayLoader
from tango.unix import ProcessDriver, ProcessForkDriver, SharedMemoryObject
from tango.webui import WebRenderer, WebDataLoader
from tango.common import ComponentType, ComponentOwner
from tango.exceptions import LoadedException

from matplotlib.figure import Figure
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.colors import LogNorm
import seaborn as sns

from abc import ABC, abstractmethod
from typing       import Sequence, Callable, Optional
from collections  import OrderedDict
from functools    import cache, cached_property, partial
from uuid         import uuid4
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
                          byref)
import numpy as np
import io
import json
import asyncio
import sys
import os

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
            'tracker.track_heat']):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['tracker'].get('type') == 'coverage'

    def __init__(self, *, driver: CoverageDriver, native_lib=None,
            verify_raw_coverage: bool=False, track_heat: bool=False, **kwargs):
        super().__init__(**kwargs)
        self._driver = driver
        self._verify = verify_raw_coverage
        self._track_heat = track_heat

        if native_lib:
            self._bind_lib = CDLL(native_lib)
        elif (lib := os.getenv("TANGO_LIBDIR")):
            self._bind_lib = CDLL(os.path.join(lib, 'tango', 'pyfeaturemap.so'))
        else:
            self._bind_lib = None

        # session-unique shm file
        self._shm_uuid = os.getpid()
        self._features_path = f'/tango_cov_{self._shm_uuid}'

    async def finalize(self, owner: ComponentOwner):
        generator = owner['generator']
        startup = getattr(generator, 'startup_input', None)

        await self._driver.relaunch()
        if startup:
            await self._driver.execute_input(startup)

        self._features = SharedMemoryObject(
            self._features_path, lambda s: b * (s // sizeof(b)))
        info(f"Obtained coverage map {self._features._size=}")

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
            return next_state
        else:
            if source:
                self._global.revert(source._feature_mask, source._hash)
            return source

    def peek(self, default_source: FeatureSnapshot=None, expected_destination: FeatureSnapshot=None, **kwargs) -> FeatureSnapshot:
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
        capture_components={'tracker'}):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['tracker'].get('type') == 'coverage'

    def __init__(self, *, tracker: CoverageTracker, **kwargs):
        super().__init__(tracker=tracker, **kwargs)

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