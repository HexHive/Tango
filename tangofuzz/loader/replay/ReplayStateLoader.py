from .. import warning

from typing       import Union
from common       import StabilityException, StateNotReproducibleException
from loader       import Environment
from input        import InputBase
from networkio    import (ChannelFactoryBase,
                         ChannelBase)
from statemanager import (StateBase,
                         StateManager)
from loader       import StateLoaderBase
from time         import sleep
from ptrace.binding import ptrace_traceme
import subprocess
from profiler import ProfileCount
from itertools import chain
from pyroute2 import netns, NetNS
import os
from uuid import uuid4

class ReplayStateLoader(StateLoaderBase):
    PROC_TERMINATE_RETRIES = 5
    PROC_TERMINATE_WAIT = 0.1 # seconds

    def __init__(self, exec_env: Environment, ch_env: ChannelFactoryBase,
            no_aslr: bool, startup_input: InputBase, path_retry_limit: int=50):
        # initialize the base class
        super().__init__(exec_env, ch_env, no_aslr)
        self._pobj = None # Popen object of child process
        self._limit = path_retry_limit
        self._startup_input = startup_input
        self._netns_name = f'ns:{uuid4()}'
        self._netns = NetNS(self._netns_name, flags=os.O_CREAT | os.O_EXCL)
        self._netns.link('set', index=1, state='up')

    def __del__(self):
        if hasattr(self, '_netns'):
            self._netns.remove()

    def _prepare_process(self):
        netns.setns(self._netns_name)
        ptrace_traceme()

    def _launch_target(self):
        # TODO later replace this by a forkserver to reduce reset costs

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

    @property
    def channel(self):
        return self._channel

    def load_state(self, state_or_path: Union[StateBase, list], sman: StateManager, update: bool = True):
        if state_or_path is None or sman is None:
            # special case where the state tracker wants an initial state
            path_gen = ((),)
        elif isinstance((state := state_or_path), StateBase):
            # WARN this path seems to lead the fuzzer to always fail to
            # reproduce some states. Disabled temporarily pending further
            # troubleshooting.
            #
            # Get a path to the target state (may throw if state not in sm)
            # path_gen = chain((sman.state_machine.get_any_path(state),),
                             # sman.state_machine.get_min_paths(state))
            path_gen = sman.state_machine.get_min_paths(state)
        else:
            path_gen = (state_or_path,)
            state = None

        # try out all paths until one succeeds
        paths_tried = 0
        exhaustive = False
        faulty_state = None
        while True:
            if exhaustive and state is not None:
                path_gen = sman.state_machine.get_paths(state)
            for path in path_gen:
                # relaunch the target and establish channel
                self._launch_target()

                ## Send startup input
                self.execute_input(self._startup_input, None, update=False)

                if sman is not None and update:
                    # FIXME should this be done here? (see comment in StateManager.reset_state)
                    sman._last_state = sman._tracker.entry_state

                paths_tried += 1
                try:
                    # reconstruct target state by replaying inputs
                    for source, destination, input in path:
                        # check if source matches the current state
                        if source != sman.state_tracker.current_state:
                            faulty_state = source
                            raise StabilityException(
                                f"source state ({source}) did not match current state ({sman.state_tracker.current_state})"
                            )
                        # perform the input
                        self.execute_input(input, sman, update=update)
                        # check if destination matches the current state
                        if destination != sman.state_tracker.current_state:
                            faulty_state = destination
                            raise StabilityException(
                                f"destination state ({destination}) did not match current state ({sman.state_tracker.current_state})"
                            )
                    break
                except StabilityException as ex:
                    warning(f"Failed to follow unstable path (reason = {ex.args[0]})! Retrying... ({paths_tried = })")
                    ProfileCount('paths_failed')(1)
                    continue
                except Exception as ex:
                    warning(f"Exception encountered following path ({ex = })! Retrying... ({paths_tried = })")
                    ProfileCount('paths_failed')(1)
                    continue
                finally:
                    if self._limit and paths_tried >= self._limit:
                        raise StateNotReproducibleException(
                            "destination state not reproducible",
                            faulty_state
                        )
            else:
                if exhaustive or state is None:
                    raise StateNotReproducibleException(
                        "destination state not reproducible",
                        faulty_state
                    )
                elif paths_tried > 0:
                    exhaustive = True
                    continue
            break