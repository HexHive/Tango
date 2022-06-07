from networkio import ChannelBase, ChannelFactoryBase
from   subprocess  import Popen
from   dataclasses import dataclass
from pyroute2.netns import setns
import socket
from os import getpid
from common import async_property

class NetworkChannel(ChannelBase):
    def __init__(self, netns: str, timescale:float, **kwargs):
        super().__init__(**kwargs)
        self._netns = netns
        self._ctx = NetNSContext(nsname=self._netns)
        self._timescale = timescale

    def nssocket(self, *args):
        """
        This is a wrapper for socket.socket() that creates the socket inside the
        specified network namespace.
        """
        with self._ctx:
            s = socket.socket(*args)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s

    @async_property
    async def timescale(self):
        return self._timescale

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

@dataclass
class NetworkChannelFactory(ChannelFactoryBase):
    timescale: float

@dataclass
class TransportChannelFactory(NetworkChannelFactory):
    endpoint: str
    port: int
