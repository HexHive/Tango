from __future__   import annotations
from . import debug, info, warning

from tango.core import (BaseState, BaseTracker, AbstractInput, LambdaProfiler,
    ValueMeanProfiler, CountProfiler, Path, BaseExplorer, LoadableTarget,
    BaseInput)
from tango.replay import ReplayLoader
from tango.unix import ProcessDriver, ProcessForkDriver
from tango.common import ComponentType, ComponentOwner
from tango.exceptions import LoadedException

from abc import ABC, abstractmethod
from typing       import Sequence, Callable, Optional
from collections  import OrderedDict
from functools    import cache, cached_property
from string import ascii_letters, digits
from uuid         import uuid4
from ctypes       import pythonapi, memset, memmove, addressof, sizeof, CDLL
from ctypes       import (POINTER as P,
                          c_bool as B,
                          c_ubyte as b,
                          c_uint32 as I,
                          c_size_t as S,
                          c_void_p as V,
                          c_char as c,
                          cast as C,
                          c_int,
                          byref, Array)
import numpy as np
import sys
import os
import mmap
import posix_ipc
import numpy as np

memcmp = pythonapi.memcmp
memcmp.argtypes = (V, V, S)
memcmp.restype = c_int

__all__ = [
    'FeatureSnapshot', 'CoverageReader', 'FeatureMap', 'CoverageTracker',
    'CoverageDriver', 'CoverageForkDriver'
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
        self._raw_coverage = feature_map._reader.clone_array()

    def __eq__(self, other):
        # return hash(self) == hash(other)
        return isinstance(other, FeatureSnapshot) and \
               hash(self) == hash(other) and \
               self._feature_count == other._feature_count
               # np.array_equal(self._feature_mask, other._feature_mask) and \
               # self._feature_context == other._feature_context

    def __repr__(self):
        return f'({self._id}) +{self._feature_count}'

class CoverageReader:
    valid_chars = frozenset("-_. %s%s" % (ascii_letters, digits))

    def __init__(self, tag, size_tag, create=False, force=False):
        # default vals so __del__ doesn't fail if __init__ fails to complete
        self._mem = None
        self._map = None
        self._owner = create

        size_tag = self.ensure_tag(size_tag)

        # get the size of the coverage array
        _type = I
        _size = sizeof(_type)
        # FIXME this is prone to race conditions when launching multiple fuzzers
        _mem, _map = self.init_array(size_tag, _type, _size, False, False)
        self._length = _type.from_address(self.address_of_buffer(_map)).value
        _map.close()
        _mem.unlink()

        info(f"Obtained coverage map {self._length=}")

        tag = self.ensure_tag(tag)

        self._type = b * self._length
        self._size = sizeof(self._type)

        self._mem, self._map = self.init_array(tag, self._type, self._size,
            create, force)
        self._array = self._type.from_address(self.address_of_buffer(self._map))

    @classmethod
    def ensure_tag(cls, tag):
        assert frozenset(tag[1:]).issubset(cls.valid_chars)
        if tag[0] != "/":
            tag = "/%s" % (tag,)
        return tag

    @staticmethod
    def init_array(tag, typ, size, create, force):
        # assert 0 <= size < sys.maxint
        assert 0 <= size < sys.maxsize
        flag = (0, posix_ipc.O_CREX)[create]
        try:
            _mem = posix_ipc.SharedMemory(tag, flags=flag, size=size)
        except posix_ipc.ExistentialError:
            if force:
                posix_ipc.unlink_shared_memory(tag)
                _mem = posix_ipc.SharedMemory(tag, flags=flag, size=size)
            else:
                raise

        _map = mmap.mmap(_mem.fd, _mem.size)
        _mem.close_fd()

        return _mem, _map

    @staticmethod
    def address_of_buffer(buf):
        return addressof(c.from_buffer(buf))

    def clone_array(self, onto: Optional[Array]=None):
        if onto is None:
            onto = self._type()
        memmove(onto, self._array, self._size)
        return onto

    def write_array(self, data: Array):
        memmove(self._array, data, self._size)

    @property
    def array(self):
        return self._array

    @property
    def length(self):
        return self._length

    @property
    def size(self):
        return self._size

    @property
    def ctype(self):
        return self._type

    @property
    def address(self):
        return self._map

    def __del__(self):
        if self._map is not None:
            self._map.close()
        if self._mem is not None and self._owner:
            self._mem.unlink()


class FeatureMap(ABC):
    def __new__(cls, *args, **kwargs):
        if (bind_lib := kwargs.get('bind_lib')):
            return super(FeatureMap, CFeatureMap).__new__(CFeatureMap)
        else:
            return super(FeatureMap, NPFeatureMap).__new__(NPFeatureMap)

    def __init__(self, reader: CoverageReader, **kwargs):
        self._reader = reader
        self._length = reader.length
        self._shared_map = reader.array

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
        if self._length != other._length:
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
        self._feature_arr = (b * self._length)()
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
        feature_mask = (b * self._length)()
        feature_count = I()
        mask_hash = I()
        coverage_map = self._shared_map
        res = self._bind_lib.diff(self._feature_arr, coverage_map, feature_mask,
            self._length, byref(feature_count), byref(mask_hash), commit)
        return feature_mask, feature_count.value, mask_hash.value

    def commit(self, feature_mask: Sequence, mask_hash: int):
        self._bind_lib.apply(self._feature_arr, feature_mask, self._length)

    def revert(self, feature_mask: Sequence, mask_hash: int):
        self._bind_lib.apply(self._feature_arr, feature_mask, self._length)

    def reset(self, feature_mask: Sequence, mask_hash: int):
        self._bind_lib.reset(self._feature_arr, feature_mask, self._length)

    def clone(self) -> CFeatureMap:
        cpy = self.__class__(self._reader, bind_lib=self._bind_lib)
        cpy.copy_from(self)
        return cpy

    def copy_from(self, other: CFeatureMap):
        super().copy_from(other)
        memmove(self._feature_arr, other._feature_arr, self._length)

    def clear(self):
        memset(self._feature_arr, 0, self._length)

    def __eq__(self, other):
        return super().__eq__(other) and \
            memcmp(self._feature_arr, other._feature_arr, self._length) == 0

class NPFeatureMap(FeatureMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._feature_arr = np.zeros(self._length, dtype=np.uint8)
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
        cpy = self.__class__(self._reader)
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
                    memset(self._tracker._reader.array, 0,
                        self._tracker._reader.size)
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
        capture_paths=['tracker.native_lib', 'tracker.verify_raw_coverage']):
    def __init__(self, *, driver: CoverageDriver, native_lib=None,
            verify_raw_coverage: bool=False, **kwargs):
        super().__init__(**kwargs)
        self._driver = driver
        self._verify = verify_raw_coverage

        if native_lib:
            self._bind_lib = CDLL(native_lib)
        elif (lib := os.getenv("TANGO_LIBDIR")):
            self._bind_lib = CDLL(os.path.join(lib, "coverage.so"))
        else:
            self._bind_lib = None

        # session-unique shm file
        self._shm_uuid = uuid4()
        self._shm_name = f'/tango_cov_{self._shm_uuid}'
        self._shm_size_name = f'/tango_size_{self._shm_uuid}'

        # set environment variables and load program with loader
        self._driver._exec_env.env.update({
            'TANGO_COVERAGE': self._shm_name,
            'TANGO_SIZE': self._shm_size_name
        })

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['tracker'].get('type') == 'coverage'

    async def finalize(self, owner: ComponentOwner):
        generator = owner['generator']
        startup = getattr(generator, 'startup_input', None)

        await self._driver.relaunch()
        if startup:
            await self._driver.execute_input(startup)

        self._reader = CoverageReader(self._shm_name, self._shm_size_name)

        # initialize feature maps
        self._global = FeatureMap(self._reader, bind_lib=self._bind_lib)
        self._scratch = FeatureMap(self._reader, bind_lib=self._bind_lib)
        self._local = FeatureMap(self._reader, bind_lib=self._bind_lib)

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
        return self.peek(self._current_state)

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
            real_cov = np.asarray(self._reader.array)
            state_cov = np.asarray(state._raw_coverage)
            if not np.array_equal(real_cov, state_cov):
                raise RuntimeError("State coverage did not match actual map.")

        # reset local map
        self._local.clear()
        self._local_state = None
        # update the local map with the latest coverage readings
        self._update_local()

    def _update_local(self):
        # this is a pseudo-state that stores the last observed diffs in the local map
        next_state = self.extract_snapshot(self._local, self._local_state,
            allow_empty=True, do_not_cache=True)
        self._local_state = next_state

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
            self._tracker._reader.write_array(dst._parent._raw_coverage)

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