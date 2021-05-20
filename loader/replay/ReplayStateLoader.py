from common       import StabilityException
from fuzzer       import (Environment,
                         ChannelFactoryBase,
                         ChannelBase)
from statemanager import (StateBase,
                         StateManager,
                         StateLoaderBase)
import subprocess
from time         import sleep

class ReplayStateLoader(StateLoaderBase):
    PROC_TERMINATE_RETRIES = 5
    PROC_TERMINATE_WAIT = 0.100 # seconds
    CHAN_CREATE_RETRIES = 5
    CHAN_CREATE_WAIT = 0.100 # seconds

    def __init__(self, exec_env: Environment, ch_env: ChannelFactoryBase):
        # initialize the base class
        super().__init__(exec_env, ch_env)
        self._pobj = None # Popen object of child process

    def _launch_target(self):
        # kill current process, if any
        if self._pobj:
            retries = 0
            while self._pobj.poll() is None:
                if retries == PROC_TERMINATE_RETRIES:
                    # TODO add logging to indicate force kill
                    self._pobj.kill()
                    break
                self._pobj.terminate()
                sleep(PROC_TERMINATE_WAIT)
                retries += 1

        # launch new process
        self._pobj = subprocess.Popen(self._exec_env.args, shell=False,
            stdin  = self._exec_env.stdin,
            stdout = self._exec_env.stdout,
            stderr = self._exec_env.stderr,
            cwd = self._exec_env.cwd,
            restore_signals = True, # TODO check if this should be false
            env = self._exec_env.env,
        )

        # establish a connection
        retries = 0
        while True:
            try:
                self._channel = self._ch_env.create()
                break
            except:
                retries += 1
                if retries == CHAN_CREATE_RETRIES:
                    raise
                sleep(CHAN_CREATE_WAIT)

    @property
    def channel(self):
        return self._channel

    def load_state(self, state: StateBase, sman: StateManager):
        # get a path to the target state (may throw if state not in sm)
        # TODO how to select from multiple paths?
        path = next(sman.state_machine.get_paths(state))

        # relaunch the target and establish channel
        self._launch_target()

        # reconstruct target state by replaying inputs
        for source, destination, transition in path:
            # check if source matches the current state
            if source != sman.state_tracker.current_state:
                raise StabilityException(
                    "source state did not match current state"
                )
            # perform the transition
            self.execute_input(transition.input, self._channel, sman)
            # check if destination matches the current state
            if destination != sman.state_tracker.current_state:
                raise StabilityException(
                    "destination state did not match current state"
                )