from tango.core import (AbstractInput, BaseMutator,
    AbstractInstruction, TransmitInstruction, ReceiveInstruction,
    DelayInstruction, BaseInputGenerator, AbstractState)
from . import debug, info, warning, critical

from array import array
from typing import Sequence, Iterable, Optional, ByteString
from random import Random
from enum import Enum
from copy import deepcopy
from random import Random
from itertools import pairwise

__all__ = [
    'HavocMutator', 'RandomInputGenerator', 'MutatedTransmitInstruction',
    'havoc_handlers'
]

"""
These mutator functions were obtained from kAFL's fuzzer.technique.helper
module. Changes were made to adapt the mutators for use with TangoFuzz.
"""

MUT_MAX_DATA_LENGTH = 1 << 15
MUT_HAVOC_BLK_LARGE = 1500
MUT_ARITH_MAX = 35
MUT_HAVOC_MIN = 2000
MUT_HAVOC_CYCLES = 5000
MUT_HAVOC_STACK_POW2 = 3# 7

interesting_8_Bit = [128, 255, 0, 1, 16, 32, 64, 100, 127]
interesting_16_Bit = [65535, 32897, 128, 255, 256, 512, 1000, 1024, 4096, 32767]
interesting_32_Bit = [4294967295, 2248146693, 2147516417, 32768, 65535, 65536, 100663045, 2147483647]

def MIN(value_a, value_b):
    if value_a > value_b:
        return value_b
    else:
        return value_a

def RAND(value, entropy):
    if value == 0:
        return 0
    return entropy.randint(0, value - 1)

def load_8(value, pos):
    return value[pos]


def load_16(value, pos):
    return (value[pos] << 8) + (value[pos+1] % 0xff)


def load_32(value, pos):
    return (value[pos] << 24) + (value[pos+1] << 16) + (value[pos+2] << 8) + (value[pos+3] % 0xff)


def store_8(data, pos, value):
    data[pos] = in_range_8(value)


def store_16(data, pos, value):
    value = in_range_16(value)
    data[pos]   = (value & 0xff00) >> 8
    data[pos+1] = (value & 0x00ff)


def store_32(data, pos, value):
    value = in_range_32(value)
    data[pos]   = (value & 0xff000000) >> 24
    data[pos+1] = (value & 0x00ff0000) >> 16
    data[pos+2] = (value & 0x0000ff00) >> 8
    data[pos+3] = (value & 0x000000ff)


def in_range_8(value):
    return value & 0xff


def in_range_16(value):
    return value & 0xffff


def in_range_32(value):
    return value & 0xffffffff


def swap_16(value):
    return (((value & 0xff00) >> 8) +
            ((value & 0xff) << 8))


def swap_32(value):
    return ((value & 0x000000ff) << 24) + \
           ((value & 0x0000ff00) << 8) + \
           ((value & 0x00ff0000) >> 8) + \
           ((value & 0xff000000) >> 24)


def bytes_to_str_8(value):
    return chr((value & 0xff))


def bytes_to_str_16(value):
    return chr((value & 0xff00) >> 8) + \
           chr((value & 0x00ff))


def bytes_to_str_32(value):
    return chr((value & 0xff000000) >> 24) + \
           chr((value & 0x00ff0000) >> 16) + \
           chr((value & 0x0000ff00) >> 8) + \
           chr((value & 0x000000ff))


def to_string_16(value):
    return chr((value >> 8) & 0xff) + \
           chr(value & 0xff)


def to_string_32(value):
    return chr((value >> 24) & 0xff) + \
           chr((value >> 16) & 0xff) + \
           chr((value >> 8) & 0xff) + \
           chr(value & 0xff)


def is_not_bitflip(value):
    return True
    if value == 0:
        return False

    sh = 0
    while (value & 1) == 0:
        sh += 1
        value >>= 1

    if value == 1 or value == 3 or value == 15:
        return False

    if (sh & 7) != 0:
        return True

    if value == 0xff or value == 0xffff or value == 0xffffffff:
        return False

    return True


def is_not_arithmetic(value, new_value, num_bytes, set_arith_max=None):
    if value == new_value:
        return False

    if not set_arith_max:
        set_arith_max = MUT_ARITH_MAX

    diffs = 0
    ov = 0
    nv = 0
    for i in range(num_bytes):
        a = value >> (8 * i)
        b = new_value >> (8 * i)
        if a != b:
            diffs += 1
            ov = a
            nv = b

    if diffs == 1:
        if in_range_8(ov - nv) <= set_arith_max or in_range_8(nv-ov) <= set_arith_max:
            return False

    if num_bytes == 1:
        return True

    diffs = 0
    for i in range(num_bytes / 2):
        a = value >> (16 * i)
        b = new_value >> (16 * i)

        if a != b:
            diffs += 1
            ov = a
            nv = b

    if diffs == 1:
        if in_range_16(ov - nv) <= set_arith_max or in_range_16(nv - ov) <= set_arith_max:
            return False

        ov = swap_16(ov)
        nv = swap_16(nv)

        if in_range_16(ov - nv) <= set_arith_max or in_range_16(nv - ov) <= set_arith_max:
            return False

    if num_bytes == 4:
        if in_range_32(value - new_value) <= set_arith_max or in_range_32(new_value - value) <= set_arith_max:
            return False

        value = swap_32(value)
        new_value = swap_32(new_value)

        if in_range_32(value - new_value) <= set_arith_max or in_range_32(new_value - value) <= set_arith_max:
            return False

    return True


def is_not_interesting(value, new_value, num_bytes, le):
    if value == new_value:
        return False

    for i in range(num_bytes):
        for j in range(len(interesting_8_Bit)):
            tval = (value & ~(0xff << (i * 8))) | (interesting_8_Bit[j] << (i * 8))
            if new_value == tval:
                return False

    if num_bytes == 2 and not le:
        return True

    for i in range(num_bytes - 1):
        for j in range(len(interesting_16_Bit)):
            tval = (value & ~(0xffff << (i * 8)) | (interesting_16_Bit[j] << (i * 8)))
            #print(" -> " + str(value) + " - " + str(new_value) + " - " + str(tval))
            if new_value == tval:
                return False

            #if num_bytes > 2:
            tval = (value & ~(0xffff << (i * 8))) | (swap_16(interesting_16_Bit[j]) << (i * 8));
            if new_value == tval:
                return False

    if num_bytes == 4 and le:
        for j in range(len(interesting_32_Bit)):
            if new_value == interesting_32_Bit[j]:
                return False

    return True

"""
These mutator functions were obtained from kAFL's fuzzer.technique.havoc_handler
module. Changes were made to adapt the mutators for use with TangoFuzz.
"""

def havoc_perform_bit_flip(data, entropy):
    if len(data) >= 1:
        bit = RAND(len(data) << 3, entropy)
        data[bit//8] ^= 1 << (bit % 8)
    return data


def havoc_perform_insert_interesting_value_8(data, entropy):
    if len(data) >= 1:
        data[RAND(len(data), entropy)] = interesting_8_Bit[RAND(len(interesting_8_Bit), entropy)]
    return data


def havoc_perform_insert_interesting_value_16(data, entropy):
    if len(data) >= 2:
        pos = RAND(len(data)-1, entropy)
        interesting_value = interesting_16_Bit[RAND(len(interesting_16_Bit), entropy)]
        if RAND(2, entropy) == 1:
            interesting_value = swap_16(interesting_value)
        store_16(data, pos, interesting_value)
    return data


def havoc_perform_insert_interesting_value_32(data, entropy):
    if len(data) >= 4:
        pos = RAND(len(data)-3, entropy)
        interesting_value = interesting_32_Bit[RAND(len(interesting_32_Bit), entropy)]
        if RAND(2, entropy) == 1:
            interesting_value = swap_32(interesting_value)
        store_32(data, pos, interesting_value)
    return data


def havoc_perform_byte_subtraction_8(data, entropy):
    if len(data) >= 1:
        pos = RAND(len(data), entropy)
        value = load_8(data, pos)
        value -= 1 + RAND(MUT_ARITH_MAX, entropy)
        store_8(data, pos, value)
    return data


def havoc_perform_byte_addition_8(data, entropy):
    if len(data) >= 1:
        pos = RAND(len(data), entropy)
        value = load_8(data, pos)
        value += 1 + RAND(MUT_ARITH_MAX, entropy)
        store_8(data, pos, value)
    return data


def havoc_perform_byte_subtraction_16(data, entropy):
    if len(data) >= 2:
        pos = RAND(len(data)-1, entropy)
        value = load_16(data, pos)
        if RAND(2, entropy) == 1:
            value = swap_16(swap_16(value) - (1 + RAND(MUT_ARITH_MAX, entropy)))
        else:
            value -= 1 + RAND(MUT_ARITH_MAX, entropy)
        store_16(data, pos, value)
    return data


def havoc_perform_byte_addition_16(data, entropy):
    if len(data) >= 2:
        pos = RAND(len(data)-1, entropy)
        value = load_16(data, pos)
        if RAND(2, entropy) == 1:
            value = swap_16(swap_16(value) + (1 + RAND(MUT_ARITH_MAX, entropy)))
        else:
            value += 1 + RAND(MUT_ARITH_MAX, entropy)
        store_16(data, pos, value)
    return data


def havoc_perform_byte_subtraction_32(data, entropy):
    if len(data) >= 4:
        pos = RAND(len(data)-3, entropy)
        value = load_32(data, pos)
        if RAND(2, entropy) == 1:
            value = swap_32(swap_32(value) - (1 + RAND(MUT_ARITH_MAX, entropy)))
        else:
            value -= 1 + RAND(MUT_ARITH_MAX, entropy)
        store_32(data, pos, value)
    return data


def havoc_perform_byte_addition_32(data, entropy):
    if len(data) >= 4:
        pos = RAND(len(data)-3, entropy)
        value = load_32(data, pos)
        if RAND(2, entropy) == 1:
            value = swap_32(swap_32(value) + (1 + RAND(MUT_ARITH_MAX, entropy)))
        else:
            value += 1 + RAND(MUT_ARITH_MAX, entropy)
        store_32(data, pos, value)
    return data


def havoc_perform_set_random_byte_value(data, entropy):
    if len(data) >= 1:
        data[RAND(len(data), entropy)] = 1 + RAND(0xff, entropy)
    return data

def havoc_perform_delete_random_byte(data, entropy):
    if len(data) >= 2:
        del_length = entropy.randrange(1, len(data))
        del_from = RAND(len(data) - del_length + 1, entropy)
        data = data[:del_from] + data[del_from + del_length:]
    return data


def havoc_perform_clone_random_byte(data, entropy):
    if len(data) >= 1 and (len(data) + MUT_HAVOC_BLK_LARGE) < MUT_MAX_DATA_LENGTH:
        clone_length = min(RAND(len(data), entropy) + 1, MUT_HAVOC_BLK_LARGE)
        clone_from = RAND(len(data) - clone_length + 1, entropy)
        clone_to = RAND(len(data), entropy)
        head = data[:clone_to]

        if RAND(4, entropy) != 0: # 75% chance existing data
            body = data[clone_from:clone_from+clone_length]
        else: # 25% chance new data
            body = bytes(entropy.randint(0, 0xff) for _ in range(clone_length))

        tail = data[clone_to:]
        data = head + body + tail
    return data


def havoc_perform_byte_seq_override(data, entropy):
    if len(data) >= 2:
        copy_length = entropy.randrange(1, len(data))
        copy_from = RAND(len(data) - copy_length + 1, entropy)
        copy_to = RAND(len(data) - copy_length + 1, entropy)
        if RAND(4, entropy) != 0: # 75% chance duplicate data
            if copy_from != copy_to:
                data[copy_to:copy_to+copy_length] = data[copy_from:copy_from + copy_length]
        else: # 25% chance duplicate arbitrary byte
            value = RAND(0xff, entropy)
            for i in range(copy_length):
                data[i+copy_to] = value
    return data


def havoc_perform_byte_seq_extra1(data):
    pass


def havoc_perform_byte_seq_extra2(data):
    pass

# TODO
def havoc_splicing(data, files=None):

    if len(data) >= 2:
        for file in files:
            file_data = read_binary_file(file)
            if len(file_data) < 2:
                continue

            first_diff, last_diff = find_diffs(data, file_data)
            if last_diff < 2 or first_diff == last_diff:
                continue

            split_location = first_diff + RAND(last_diff - first_diff, entropy)

            data = array('B', data.tostring()[:split_location] + file_data[split_location:len(data)])
            break

    return data

dict_import = []
def set_dict(new_dict):
    global dict_import
    dict_import = new_dict

def append_handler(handler):
    global havoc_handlers
    havoc_handlers.append(handler)

def havoc_dict(data, entropy):
    global dict_import
    if len(dict_import) > 0:
        dict_entry = dict_import[RAND(len(dict_import), entropy)]
        dict_entry = dict_entry[:len(data)]
        entry_pos = RAND(len(data)-len(dict_entry)+1, entropy)
        data = data[:entry_pos] + dict_entry + data[entry_pos + len(dict_entry):]
    return data

havoc_handlers = [havoc_perform_bit_flip,
                  havoc_perform_insert_interesting_value_8,
                  havoc_perform_insert_interesting_value_16,
                  havoc_perform_insert_interesting_value_32,
                  havoc_perform_byte_addition_8,
                  havoc_perform_byte_addition_16,
                  havoc_perform_byte_addition_32,
                  havoc_perform_byte_subtraction_8,
                  havoc_perform_byte_subtraction_16,
                  havoc_perform_byte_subtraction_32,
                  havoc_perform_set_random_byte_value,
                  havoc_perform_delete_random_byte,
                  havoc_perform_clone_random_byte,
                  havoc_perform_byte_seq_override,
                  #havoc_perform_byte_seq_extra1,
                  #havoc_perform_byte_seq_extra2,
                 ]

class MutatedTransmitInstruction(TransmitInstruction):
    def __init__(self, data: ByteString, transforms: Iterable):
        super().__init__(data)
        self.transforms = transforms

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__init__(self._data, self.transforms)
        return result

class HavocMutator(BaseMutator):
    class RandomOperation(Enum):
        DELETE = 0
        PUSHORDER = 1
        POPORDER = 2
        REPEAT = 3
        CREATE = 4
        MUTATE = 5

    class RandomInstruction(Enum):
        TRANSMIT = 0
        RECEIVE = 1
        DELAY = 2

    def __init__(self, input: AbstractInput, /, *,
            weights: Optional[Sequence[float]]=None,
            k: Optional[int]=None,
            havoc_actions: Optional[Sequence]=None,
            chunk_size: Optional[int]=None,
            **kwargs):
        super().__init__(input, **kwargs)
        k = k or RAND(MUT_HAVOC_STACK_POW2, self._entropy) + 1
        self._actions = havoc_actions or self._entropy.choices(
                havoc_handlers, weights=weights, k=k)
        self._chunk = chunk_size

    def __iter__(self, *, orig):
        with self.entropy_ctx as entropy:
            i = -1
            reorder_buffer = []
            for i, instruction in enumerate(orig()):
                info(f"Mutating {instruction}")
                seq = self._mutate(instruction, reorder_buffer, entropy)
                yield from seq
            if i == -1:
                info(f"Mutating but there is no instruction")
                yield from self._mutate(None, reorder_buffer, entropy)

            # finally, we flush the reorder buffer
            info(f"Flushing the reorder buffer")
            yield from entropy.sample(reorder_buffer, k=len(reorder_buffer))
            reorder_buffer.clear()

    def __repr__(self, *, orig):
        return f'HavocMutatedInput:0x{self.id:08X} (0x{self._orig.id:08X})'

    def _mutate(self, instruction: AbstractInstruction, reorder_buffer: Sequence, entropy: Random) -> Sequence[AbstractInstruction]:
        if instruction is not None:
            low = 0
            for _ in range(entropy.randint(3, 7)):
                if low > 5:
                    return
                oper = self.RandomOperation(entropy.randint(low, 5))
                low = oper.value + 1
                info(f"Applying mutator {oper}")
                if oper == self.RandomOperation.DELETE:
                    return
                elif oper == self.RandomOperation.PUSHORDER:
                    reorder_buffer.append(deepcopy(instruction))
                    return
                elif oper == self.RandomOperation.POPORDER:
                    yield deepcopy(instruction)
                    if reorder_buffer:
                        yield from self._mutate(reorder_buffer.pop(), reorder_buffer, entropy)
                elif oper == self.RandomOperation.REPEAT:
                    yield from (deepcopy(instruction) for _ in range(2))
                elif oper == self.RandomOperation.CREATE:
                    buffer = entropy.randbytes(entropy.randint(1, 256))
                    for chunk in self._chunk_up(buffer):
                        yield TransmitInstruction(chunk)
                elif oper == self.RandomOperation.MUTATE:
                    if isinstance(instruction, TransmitInstruction):
                        mut_data = self._apply_actions(instruction._data, entropy)
                        for chunk in self._chunk_up(mut_data):
                            yield MutatedTransmitInstruction(chunk, self._actions)
                    else:
                        # no mutations on other instruction types for now,
                        # and we drop them
                        pass
        else:
            buffer = entropy.randbytes(entropy.randint(1, 256))
            for chunk in self._chunk_up(buffer):
                yield TransmitInstruction(chunk)

    def _chunk_up(self, data):
        if not self._chunk:
            yield data
        else:
            aligned = (len(data) - 1) // self._chunk + 1
            aligned *= self._chunk
            ranges = pairwise(range(0, aligned, self._chunk))
            for begin, end in ranges:
                yield data[begin:end]

    def _apply_actions(self, data, entropy):
        for func in self._actions:
            # this copies the data buffer into a new array
            data = bytearray(data)
            data = func(data, entropy)
        return data

class RandomInputGenerator(BaseInputGenerator):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['generator'].get('type') == 'random'

    def generate(self, state: AbstractState) -> AbstractInput:
        info(f"Generating input in {self}")
        candidate = self.select_candidate(state)
        debug(f"Selected candidate {candidate}")
        info(f"Mutating the candidate {candidate}")
        mut = HavocMutator(candidate, entropy=self._entropy,
            chunk_size=self._fmt.chunk_size)
        info(f"Generated mutatable input {mut}")
        return mut
