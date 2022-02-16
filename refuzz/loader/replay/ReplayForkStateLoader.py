from .. import warning

from common       import StabilityException, StateNotReproducibleException
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
from profiler import ProfileCount

class ReplayForkStateLoader(StateLoaderBase):
    def __init__(self, exec_env: Environment, ch_env: ChannelFactoryBase,
            no_aslr: bool, path_retry_limit=50):
        # initialize the base class
        super().__init__(exec_env, ch_env, no_aslr)
        self._pobj = None # Popen object of child process
        self._limit = path_retry_limit

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
            path_gen = ((),)
        else:
            # get a path to the target state (may throw if state not in sm)
            # TODO how to select from multiple paths?
            path_gen = sman.state_machine.get_min_paths(state)

        # try out all paths until one succeeds
        paths_tried = 1
        exhaustive = False
        faulty_state = None
        while True:
            if exhaustive:
                path_gen = sman.state_machine.get_paths(state)
            for path in path_gen:
                # relaunch the target and establish channel
                self._launch_target()

                if sman is not None and update:
                    # FIXME should this be done here? (see comment in StateManager.reset_state)
                    sman._last_state = sman._tracker.entry_state
                try:
                    # reconstruct target state by replaying inputs
                    for source, destination, input in path:
                        # check if source matches the current state
                        if source != sman.state_tracker.current_state:
                            faulty_state = source
                            raise StabilityException(
                                "source state did not match current state"
                            )
                        # perform the input
                        self.execute_input(input, sman, update=update)
                        # check if destination matches the current state
                        if destination != sman.state_tracker.current_state:
                            faulty_state = destination
                            raise StabilityException(
                                "destination state did not match current state"
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
                    paths_tried += 1
                    if self._limit and paths_tried >= self._limit:
                        raise StateNotReproducibleException(
                            "destination state not reproducible",
                            faulty_state
                        )
            else:
                if exhaustive:
                    raise StateNotReproducibleException(
                        "destination state not reproducible",
                        faulty_state
                    )
                elif paths_tried > 0:
                    exhaustive = True
                    continue
            break