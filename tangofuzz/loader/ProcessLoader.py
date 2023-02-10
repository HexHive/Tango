from loader import StateLoaderBase, Environment
import ctypes
from ptrace.binding import ptrace_traceme
from pyroute2 import netns, IPRoute
import subprocess
from uuid import uuid4
import os
from common import sync_to_async, GLOBAL_ASYNC_EXECUTOR
from concurrent.futures import ThreadPoolExecutor

class ProcessLoader(StateLoaderBase):
    PROC_TERMINATE_RETRIES = 5
    PROC_TERMINATE_WAIT = 0.1 # seconds

    def __init__(self, /, *, exec_env: Environment, no_aslr: bool, **kwargs):
        super().__init__(**kwargs)
        self._exec_env = exec_env
        self._pobj = None # Popen object of child process
        self._netns_name = f'ns:{uuid4()}'

        if no_aslr:
            ADDR_NO_RANDOMIZE = 0x0040000
            personality = ctypes.CDLL(None).personality
            personality.restype = ctypes.c_int
            personality.argtypes = [ctypes.c_ulong]
            personality(ADDR_NO_RANDOMIZE)

    def __del__(self):
        netns.remove(self._netns_name)

    def _prepare_process(self):
        netns.setns(self._netns_name, flags=os.O_CREAT)
        with IPRoute() as ipr:
            ipr.link('set', index=1, state='up')
        ptrace_traceme()

    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def _launch_target(self):
        ## Kill current process, if any
        if self._pobj:
            # ensure that the channel is closed and the debugger detached
            self._channel.close(terminate=True)

            retries = 0
            while True:
                if retries == self.PROC_TERMINATE_RETRIES:
                    # TODO add logging to indicate force kill
                    # FIXME is safe termination necessary?
                    self._pobj.kill()
                    break
                self._pobj.terminate()
                try:
                    self._pobj.wait(self.PROC_TERMINATE_WAIT)
                    break
                except subprocess.TimeoutExpired:
                    retries += 1

        ## Launch new process
        self._pobj = subprocess.Popen(self._exec_env.args, shell=False,
            executable = self._exec_env.path,
            stdin  = self._exec_env.stdin,
            stdout = self._exec_env.stdout,
            stderr = self._exec_env.stderr,
            cwd = self._exec_env.cwd,
            restore_signals = True, # TODO check if this should be false
            env = self._exec_env.env,
            preexec_fn = self._prepare_process
        )

        ## Establish a connection
        self._channel = self._ch_env.create(self._pobj, self._netns_name)