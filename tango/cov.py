from __future__   import annotations
from . import debug, info

from tango.core import (AbstractState, BaseState, BaseStateTracker,
    AbstractInput, LambdaProfiler, ValueMeanProfiler, CountProfiler)
from tango.unix import ProcessDriver, ProcessForkDriver
from tango.common import ComponentType, ComponentOwner
from tango.exceptions import LoadedException

from abc import ABC, abstractmethod
from typing       import Sequence, Callable, Optional
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
    'CoverageState', 'CoverageReader', 'GlobalCoverage', 'CoverageStateTracker',
    'CoverageDriver', 'CoverageForkDriver'
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

class CoverageDriver(ProcessDriver,
        capture_paths=('driver.clear_coverage',)):
    @classmethod
    def match_config(cls, config: dict):
        return super().match_config(config) and \
            config['tracker'].get('type') == 'coverage'

    def __init__(self, *, clear_coverage: Optional[bool]=True, **kwargs):
        super().__init__(**kwargs)
        self._clear_cov = clear_coverage

    async def finalize(self, owner: ComponentOwner):
        # WARN this bypasses the expected component hierarchy and would usually
        # result in cyclic dependencies, but since both components are defined
        # and confined within this module, they are expected to be tightly
        # coupled and be aware of this dependency
        self._tracker: CoverageStateTracker = owner['tracker']
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
        except Exception as ex:
            raise LoadedException(ex, lambda: input[:idx]) from ex
        finally:
            ValueMeanProfiler("input_len", samples=100)(idx)
            CountProfiler("total_instructions")(idx)

# this class exists only to allow matching with forkserver==true
class CoverageForkDriver(CoverageDriver, ProcessForkDriver):
    pass

class CoverageState(BaseState):
    __slots__ = '_parent', '_set_map', '_set_count', '_context'

    def __init__(self, parent: CoverageState, set_map: Sequence, set_count: int,
            global_cov: GlobalCoverage, **kwargs):
        super().__init__(**kwargs)
        self._parent = parent
        self._set_map = set_map
        self._set_count = set_count
        self._context = global_cov.clone()
        self._context.revert(set_map, hash(self))

    def __eq__(self, other):
        # return hash(self) == hash(other)
        return isinstance(other, CoverageState) and \
               hash(self) == hash(other) and \
               self._set_count == other._set_count
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
        self._size = sizeof(_type)

        self._mem, self._map = self.init_array(tag, _type, self._size, create, force)
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
    def size(self):
        return self._size

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
    def reset(self, set_map: Sequence, map_hash: int):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        return isinstance(other, GlobalCoverage)

class CGlobalCoverage(GlobalCoverage):
    def __init__(self, *args, bind_lib, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_arr = (b * self._length)()
        self.clear()

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

        self._bind_lib.reset.argtypes = (
            P(b), # global coverage array (set_arr)
            P(b), # set_map buffer
            S # common size of the coverage buffers
        )
        self._bind_lib.reset.restype = B # success?

    def update(self, coverage_map: Sequence) -> (Sequence, int, int):
        set_map = (b * self._length)()
        set_count = I()
        map_hash = I()
        res = self._bind_lib.diff(self._set_arr, coverage_map, set_map, self._length,
            byref(set_count), byref(map_hash),
            True
        )
        return set_map, set_count.value, map_hash.value

    def revert(self, set_map: Sequence, map_hash: int):
        self._bind_lib.apply(self._set_arr, set_map, self._length)

    def clone(self) -> CGlobalCoverage:
        cpy = self.__class__(self._length, bind_lib=self._bind_lib)
        cpy.copy_from(self)
        return cpy

    def copy_from(self, other: CGlobalCoverage):
        super().copy_from(other)
        memmove(self._set_arr, other._set_arr, self._length)

    def clear(self):
        memset(self._set_arr, 0, self._length)

    def reset(self, set_map: Sequence, map_hash: int):
        self._bind_lib.reset(self._set_arr, set_map, self._length)

    def __eq__(self, other):
        return super().__eq__(other) and \
            pythonapi.memcmp(self._set_arr, other._set_arr, self._length) == 0

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

    def reset(self, set_map: Sequence, map_hash: int):
        self._set_arr &= set_map

    def __eq__(self, other):
        return super().__eq__(other) and \
               np.array_equal(self._set_arr, other._set_arr)

class CoverageStateTracker(BaseStateTracker,
        capture_components={ComponentType.driver},
        capture_paths=['tracker.native_lib']):
    def __init__(self, *, driver: ProcessDriver, native_lib=None, **kwargs):
        super().__init__(**kwargs)
        self._driver = driver

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

        # initialize a global coverage map
        self._global = GlobalCoverage(self._reader.length, bind_lib=self._bind_lib)
        self._scratch = GlobalCoverage(self._reader.length, bind_lib=self._bind_lib)
        self._local = GlobalCoverage(self._reader.length, bind_lib=self._bind_lib)

        self._local_state = None
        self._current_state = None
        # the update creates a new initial _current_state
        self.update_state(None, input=None)
        self._entry_state = self._current_state

        await super().finalize(owner)

        # initialize local map and local state
        self.reset_state(self._current_state)

        LambdaProfiler('global_cov')(lambda: sum(
            map(lambda x: x._set_count,
                filter(lambda x: x != self._entry_state,
                    self._state_graph.nodes
                )
            )
        ))

    @property
    def entry_state(self) -> CoverageState:
        return self._entry_state

    @property
    def current_state(self) -> CoverageState:
        return self.peek(self._current_state)

    def _diff_global_to_state(self, global_map: GlobalCoverage, parent_state: AbstractState, *,
            do_not_cache: bool=False, allow_empty: bool=False) -> AbstractState:
        coverage_map = self._reader.array
        set_map, set_count, map_hash = global_map.update(coverage_map)
        if set_count or allow_empty:
            return CoverageState(parent_state, set_map, set_count, global_map,
                state_hash=map_hash, do_not_cache=do_not_cache, tracker=self)
        else:
            return None

    def update_state(self, source: AbstractState, /, *, input: AbstractInput,
            exc: Exception=None, peek_result: Optional[AbstractState]=None) \
            -> AbstractState:
        source = super().update_state(source, input=input, exc=exc,
                peek_result=peek_result)
        if not exc:
            if peek_result is None:
                next_state = self._diff_global_to_state(self._global,
                    parent_state=source, allow_empty=source is None)
            else:
                # if peek_result was specified, we can skip the recalculation
                next_state = peek_result
                self._global.copy_from(next_state._context)
                # we un-revert the bitmaps to obtain the actual global context
                self._global.revert(next_state._set_map, next_state._hash)

            if not next_state:
                next_state = source
            self._current_state = next_state

            # update local coverage
            self._update_local()
            return next_state
        else:
            if source:
                self._global.revert(source._set_map, source._hash)
            return source

    def peek(self, default_source: AbstractState=None, expected_destination: AbstractState=None, **kwargs) -> AbstractState:
        glbl = self._scratch
        if expected_destination:
            # when the destination is not None, we use its `context_map` as a
            # basis for calculating the coverage delta
            glbl.copy_from(expected_destination._context)
            parent = expected_destination._parent
        else:
            glbl.copy_from(self._global)
            parent = default_source

        next_state = self._diff_global_to_state(glbl, parent, allow_empty=parent is None, **kwargs)
        if not next_state:
            next_state = default_source
        return next_state

    def reset_state(self, state: AbstractState):
        super().reset_state(state)
        self._current_state = state

        # reset local map
        self._local.clear()
        self._local_state = None
        # update the local map with the latest coverage readings
        self._update_local()

    def _update_local(self):
        # this is a pseudo-state that stores the last observed diffs in the local map
        next_state = self._diff_global_to_state(self._local, self._local_state,
            do_not_cache=True, allow_empty=True)
        # TODO is there special handling needed if next_state is None?
        self._local_state = next_state
