from dataio import AbstractChannel, AbstractChannelFactory
from   subprocess  import Popen
from   dataclasses import dataclass
from pyroute2.netns import setns
import socket
from os import getpid
from common import async_property
from dataio import FormatDescriptor

class NetworkChannel(AbstractChannel):
    def __init__(self, netns: str, **kwargs):
        super().__init__(**kwargs)
        self._netns = netns
        self._ctx = NetNSContext(nsname=self._netns)

    def nssocket(self, *args):
        """
        This is a wrapper for socket.socket() that creates the socket inside the
        specified network namespace.
        """
        with self._ctx:
            s = socket.socket(*args)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s

class NetNSContext (object):
    """
    This is a context manager that on entry assigns the process to an alternate
    network namespace (specified by name, filesystem path, or pid) and then
    re-assigns the process to its original network namespace on exit.
    """
    def __init__(self, nsname=None, nspath=None, nspid=None):
        self.mypath = self.get_ns_path(nspid=getpid())
        self.targetpath = self.get_ns_path(nspath, nsname=nsname, nspid=nspid)
        if not self.targetpath:
            raise ValueError('invalid namespace')
        self.myns = open(self.mypath)
        self.targetns = open(self.targetpath)

    def __enter__(self):
        # before entering a new namespace, we open a file descriptor in the
        # current namespace that we will use to restore our namespace on exit.
        setns(self.targetns.fileno(), 0)

    def __exit__(self, *args):
        setns(self.myns.fileno(), 0)

    def __del__(self):
        if hasattr(self, 'myns'):
            self.myns.close()
            self.targetns.close()

    @staticmethod
    def get_ns_path(nspath=None, nsname=None, nspid=None):
        """
        This is just a convenience function that will return the path to an
        appropriate namespace descriptor, give either a path, a network
        namespace name, or a pid.
        """
        if nsname:
            nspath = '/var/run/netns/%s' % nsname
        elif nspid:
            nspath = '/proc/%d/ns/net' % nspid

        return nspath

# kw_only=True is needed because typ is no longer a non-default attribute
@dataclass(kw_only=True, frozen=True)
class NetworkFormatDescriptor(FormatDescriptor):
    typ: str = 'pcap'

@dataclass(frozen=True)
class _TransportFormatDescriptor(FormatDescriptor):
    """
    This class is needed since `protocol` is a non-default attribute that will
    be specified at instantiation, but the `fmt` attribute is otherwise set
    to 'pcap', and it occurs before `protocol` in the reversed MRO.
    Using this class as the last class in the MRO places `protocol` right
    after the non-default `fmt`, satisfying the requirements for a dataclass.
    """
    protocol: str

@dataclass(frozen=True)
class TransportFormatDescriptor(NetworkFormatDescriptor, _TransportFormatDescriptor):
    def __get__(self, obj, owner):
        if obj is None:
            return self
        return getattr(obj, '_fmt')

    def __set__(self, obj, value):
        fmt = type(self)(protocol=value)
        setattr(obj, '_fmt', fmt)

@dataclass
class NetworkChannelFactory(AbstractChannelFactory):
    pass

@dataclass(kw_only=True)
class TransportChannelFactory(NetworkChannelFactory):
    endpoint: str
    port: int
    protocol: str
    fmt: FormatDescriptor = TransportFormatDescriptor(protocol=None)

    def __post_init__(self):
        self.fmt = self.protocol # implicit casting through the descriptor

