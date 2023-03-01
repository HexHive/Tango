from __future__   import annotations
from . import debug, info

from tango.core import (AbstractState, BaseState, AbstractStateTracker,
    AbstractInput)
from tango.common import ComponentType
from tango.unix import ProcessLoader

from abc import ABC, abstractmethod
from typing       import Sequence, Callable
from collections  import OrderedDict
from functools    import cache
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
                          byref)
import numpy as np
import sys
import os
import mmap
import posix_ipc
import numpy as np

__all__ = [
    'CoverageState', 'CoverageReader', 'GlobalCoverage', 'CoverageStateTracker'
]

pythonapi.memcmp.argtypes = (V, V, S)

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

class CoverageState(BaseState):
    _cache = {}
    _id = 0

    def __new__(cls, parent: CoverageState, set_map: Sequence, set_count: int, map_hash: int, global_cov: GlobalCoverage, do_not_cache: bool=False, **kwargs):
        _hash = map_hash
        if not do_not_cache and (cached := cls._cache.get(_hash)):
            return cached
        new = super(CoverageState, cls).__new__(cls)
        new._parent = parent
        new._set_map = set_map
        new._set_count = set_count
        new._hash = _hash
        new._context = global_cov.clone()
        # to obtain the context from the current global map, we revert the bits
        new._context.revert(set_map, map_hash)
        if not do_not_cache:
            cls._cache[_hash] = new
            new._id = cls._id
            cls._id += 1
        else:
            new._id = '(local)'
        super(CoverageState, new).__init__(**kwargs)
        return new

    def __init__(self, parent: CoverageState, set_map: Sequence, set_count: int, map_hash: int, global_cov: GlobalCoverage, do_not_cache: bool=False, **kwargs):
        super().__init__(**kwargs)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        # return hash(self) == hash(other)
        return isinstance(other, CoverageState) and \
               hash(self) == hash(other)
               # self._set_count == other._set_count and \
               # np.array_equal(self._set_map, other._set_map) and \
               # self._context == other._context

    def __repr__(self):
        return f'({self._id}) +{self._set_count}'

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

        _type = b * self._length
        _size = sizeof(_type)

        self._mem, self._map = self.init_array(tag, _type, _size, create, force)
        self._array = _type.from_address(self.address_of_buffer(self._map))

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

    @property
    def array(self):
        return self._array

    @property
    def length(self):
        return self._length

    @property
    def address(self):
        return self._map

    def __del__(self):
        if self._map is not None:
            self._map.close()
        if self._mem is not None and self._owner:
            self._mem.unlink()


class GlobalCoverage(ABC):
    def __new__(cls, *args, **kwargs):
        if (bind_lib := kwargs.get('bind_lib')):
            return super(GlobalCoverage, CGlobalCoverage).__new__(CGlobalCoverage)
        else:
            return super(GlobalCoverage, NPGlobalCoverage).__new__(NPGlobalCoverage)

    def __init__(self, length: int, **kwargs):
        self._length = length

    @abstractmethod
    def update(self, coverage_map: Sequence) -> (Sequence, int, int):
        raise NotImplementedError

    @abstractmethod
    def revert(self, set_map: Sequence, map_hash: int):
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> GlobalCoverage:
        raise NotImplementedError

    @abstractmethod
    def copy_from(self, other: GlobalCoverage):
        if self._length != other._length:
            raise RuntimeError("Mismatching coverage map sizes")

    @abstractmethod
    def clear(self):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        return isinstance(other, GlobalCoverage)

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError

class CGlobalCoverage(GlobalCoverage):
    def __init__(self, *args, bind_lib, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_arr = (b * self._length)()
        self.clear()
        self._hash = 0

        self._bind_lib = bind_lib
        self._bind_lib.diff.argtypes = (
            P(b), # global coverage array (set_arr)
            P(b), # local coverage array (coverage_map)
            P(b), # pre-allocated set_map output buffer
            S, # common size of the coverage buffers
            P(I), # pre-allocated set_count output buffer
            P(I), # pre-allocated hash output buffer
            B, # boolean: apply in-place?
        )
        self._bind_lib.diff.restype = B # success?

        self._bind_lib.apply.argtypes = (
            P(b), # global coverage array (set_arr)
            P(b), # set_map buffer
            S # common size of the coverage buffers
        )
        self._bind_lib.apply.restype = B # success?

    def update(self, coverage_map: Sequence) -> (Sequence, int, int):
        set_map = (b * self._length)()
        set_count = I()
        map_hash = I()
        res = self._bind_lib.diff(self._set_arr, coverage_map, set_map, self._length,
            byref(set_count), byref(map_hash),
            True
        )
        self._hash ^= map_hash.value
        return set_map, set_count.value, map_hash.value

    def revert(self, set_map: Sequence, map_hash: int):
        self._bind_lib.apply(self._set_arr, set_map, self._length)
        self._hash ^= map_hash

    def clone(self) -> CGlobalCoverage:
        cpy = self.__class__(self._length, bind_lib=self._bind_lib)
        cpy.copy_from(self)
        return cpy

    def copy_from(self, other: CGlobalCoverage):
        super().copy_from(other)
        memmove(self._set_arr, other._set_arr, self._length)
        other._hash = self._hash

    def clear(self):
        memset(self._set_arr, 0, self._length)

    def __eq__(self, other):
        return super().__eq__(other) and \
            self._hash == other._hash and \
            pythonapi.memcmp(self._set_arr, other._set_arr, self._length) == 0

    def __hash__(self):
        return self._hash

class NPGlobalCoverage(GlobalCoverage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_arr = np.zeros(self._length, dtype=np.uint8)
        self.clear()

    def update(self, coverage_map: Sequence) -> (Sequence, int, int):
        coverage_map = np.ctypeslib.as_array(coverage_map)
        kls_arr = CLASS_LUT[coverage_map]
        set_map = (self._set_arr | kls_arr) ^ self._set_arr
        self._set_arr ^= set_map
        set_count = np.sum(HAMMING_LUT[set_map])
        map_hash = hash(set_map.data.tobytes())

        return set_map, set_count, map_hash

    def revert(self, set_map: Sequence, map_hash: int):
        self._set_arr ^= set_map

    def clone(self) -> NPGlobalCoverage:
        cpy = self.__class__(self._length)
        cpy.copy_from(self)
        return cpy

    def copy_from(self, other: NPGlobalCoverage):
        super().copy_from(other)
        np.copyto(self._set_arr, other._set_arr)

    def clear(self):
        self._set_arr.fill(0)

    def __eq__(self, other):
        return super().__eq__(other) and \
               np.array_equal(self._set_arr, other._set_arr)

    def __hash__(self):
        return hash(self._set_arr.data.tobytes())

class LoaderDependentTracker(AbstractStateTracker,
        capture_components={ComponentType.loader}):
    def __init__(self, *, loader: ProcessLoader, **kwargs):
        super().__init__(**kwargs)
        self._loader = loader

class CoverageStateTracker(LoaderDependentTracker,
        capture_paths=['tracker.native_lib']):
    def __init__(self, *, native_lib: Optional[str]=None, **kwargs):
        super().__init__(**kwargs)

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
        self._loader._exec_env.env.update({
            'TANGO_COVERAGE': self._shm_name,
            'TANGO_SIZE': self._shm_size_name
        })

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['tracker'].get('type') == 'coverage'

    async def initialize(self):
        await super().initialize()
        load = self._loader.load_state(None)
        try:
            await load.asend(None)
        except StopAsyncIteration:
            pass

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

    @property
    def entry_state(self) -> CoverageState:
        return self._entry_state

    @property
    def current_state(self) -> CoverageState:
        return self.peek(self._current_state)

    def _diff_global_to_state(self, global_map: GlobalCoverage, parent_state: AbstractState,
            local_state: bool=False, allow_empty: bool=False) -> AbstractState:
        coverage_map = self._reader.array
        set_map, set_count, map_hash = global_map.update(coverage_map)
        if set_count or allow_empty:
            return CoverageState(parent_state, set_map, set_count, map_hash,
                global_map, do_not_cache=local_state, tracker=self)
        else:
            return None

    def update(self, source: AbstractState, input: AbstractInput, peek_result: AbstractState=None) -> AbstractState:
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

    def peek(self, default_source: AbstractState=None, expected_destination: AbstractState=None) -> AbstractState:
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

    def reset_state(self, state: AbstractState):
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
