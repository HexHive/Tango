"""
These mutator functions were obtained from kAFL's fuzzer.technique.havoc_handler
module. Changes were made to adapt the mutators for use with TangoFuzz.
"""

from array import array
from mutator.MutatorHelpers import *

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