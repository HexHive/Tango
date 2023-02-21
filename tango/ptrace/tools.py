from tango.ptrace.ctypes_tools import (
    formatUintHex16, formatUintHex32, formatWordHex)

from ctypes import sizeof
from datetime import datetime, timedelta

def dumpRegs(log, regs):
    """
    Dump all registers using log callback (write one line).
    """
    width = max(len(name) for name, type in regs._fields_)
    name_format = "%% %us" % width
    for name, type in regs._fields_:
        value = getattr(regs, name)
        name = name_format % name
        if sizeof(type) == 32:
            value = formatUintHex32(value)
        elif sizeof(type) == 16:
            value = formatUintHex16(value)
        else:
            value = formatWordHex(value)
        log("%s = %s" % (name, value))


def readBits(value, bitmasks):
    """
    Extract bits from the integer value using a list of bit masks.
    bitmasks is a list of tuple (mask, text).

    >>> bitmask = (
    ...    (1, "exec"),
    ...    (2, "write"),
    ...    (4, "read"))
    ...
    >>> readBits(5, bitmask)
    ['exec', 'read']
    >>> readBits(12, bitmask)
    ['read', '8']
    """
    bitset = []
    for mask, item in bitmasks:
        if not value & mask:
            continue
        bitset.append(item)
        value = value & ~mask
    if value:
        bitset.append(str(value))
    return bitset


def formatBits(value, bitmasks, empty_text=None, format_value=str):
    """
    Format a value using a bitmask: see readBits() functions.

    >>> bitmask = (
    ...    (1, "exec"),
    ...    (2, "write"),
    ...    (4, "read"))
    ...
    >>> formatBits(5, bitmask, empty_text="no permission")
    '<exec|read> (5)'
    >>> formatBits(0, bitmask, empty_text="no permission")
    'no permission'
    """

    orig_value = value
    text = readBits(value, bitmasks)
    if text:
        text = "%s" % ("|".join(text))
        if value:
            text = "<%s> (%s)" % (text, format_value(orig_value))
        return text
    else:
        if empty_text:
            return empty_text
        else:
            return str(value)


LOCAL_TIMEZONE_OFFSET = datetime.fromtimestamp(
    0) - datetime.utcfromtimestamp(0)

# Start of UNIX timestamp (Epoch): 1st January 1970 at 00:00
UNIX_TIMESTAMP_T0 = datetime(1970, 1, 1)


def timestampUNIX(value, is_local):
    """
    Convert an UNIX (32-bit) timestamp to datetime object. Timestamp value
    is the number of seconds since the 1st January 1970 at 00:00. Maximum
    value is 2147483647: 19 january 2038 at 03:14:07.

    May raise ValueError for invalid value: value have to be in 0..2147483647.

    >>> timestampUNIX(0, False)
    datetime.datetime(1970, 1, 1, 0, 0)
    >>> timestampUNIX(1154175644.37, False)
    datetime.datetime(2006, 7, 29, 12, 20, 44, 370000)
    """
    timestamp = UNIX_TIMESTAMP_T0 + timedelta(seconds=value)
    if is_local:
        timestamp += LOCAL_TIMEZONE_OFFSET
    return timestamp


def minmax(min_value, value, max_value):
    """
    Restrict value to [min_value; max_value]

    >>> minmax(-2, -3, 10)
    -2
    >>> minmax(-2, 27, 10)
    10
    >>> minmax(-2, 0, 10)
    0
    """
    return min(max(min_value, value), max_value)

def signal_to_exitcode(signum):
    """
    Converts a signal number to an exit code.
    UNIX: https://www.gnu.org/software/bash/manual/html_node/Exit-Status.html

    >>> import signal
    >>> signal_to_exitcode(signal.SIGQUIT)
    131
    """
    return 128 + signum