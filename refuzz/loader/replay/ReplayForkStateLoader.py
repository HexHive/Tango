from common       import StabilityException
from loader       import Environment
from networkio    import (ChannelFactoryBase,
                         ChannelBase)
from statemanager import (StateBase,
                         StateManager)
from loader       import StateLoaderBase
from time         import sleep
from ptrace.binding import ptrace_traceme
from os import kill
from signal import SIGTERM, SIGKILL
import subprocess

class ReplayForkStateLoader(StateLoaderBase):
    def __init__(self, exec_env: Environment, ch_env: ChannelFactoryBase,
            no_aslr: bool):
        # initialize the base class
        super().__init__(exec_env, ch_env, no_aslr)
        self._pobj = None # Popen object of child process

    def _launch_target(self):
        if not self._pobj:
            ## Launch new process
            self._pobj = subprocess.Popen(self._exec_env.args, shell=False,
                executable = self._exec_env.path,
                stdin  = self._exec_env.stdin,
                stdout = self._exec_env.stdout,
                stderr = self._exec_env.stderr,
                cwd = self._exec_env.cwd,
                restore_signals = True, # TODO check if this should be false
                env = self._exec_env.env,
                preexec_fn = ptrace_traceme
            )
        elif self._channel:
            ## Kill current process, if any
            try:
                self._channel.close(terminate=True)
            except ProcessLookupError:
                pass

        ## Establish a connection
        self._channel = self._ch_env.create(self._pobj)

    @property
    def channel(self):
        return self._channel

    def load_state(self, state: StateBase, sman: StateManager, update: bool = True):
        if state is None or sman is None:
            # special case where the state tracker wants an initial state
            path = ()
        else:
            # get a path to the target state (may throw if state not in sm)
            # TODO how to select from multiple paths?
            path = next(sman.state_machine.get_paths(state))

        # relaunch the target and establish channel
        self._launch_target()

        # reconstruct target state by replaying inputs
        for source, destination, input in path:
            # check if source matches the current state
            if source != sman.state_tracker.current_state:
                raise StabilityException(
                    "source state did not match current state"
                )
            # perform the input
            self.execute_input(input, sman, update=update)
            # check if destination matches the current state
            if destination != sman.state_tracker.current_state:
                raise StabilityException(
                    "destination state did not match current state"
                )