from .. import warning

from typing       import Union
from common       import StabilityException, StateNotReproducibleException, async_wrapper
from loader       import Environment
from input        import InputBase, ZoomInput
from interaction  import (ReachInteraction, ActivateInteraction, DelayInteraction,
                         MoveInteraction)
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
import asyncio

class ZoomStateLoader(StateLoaderBase):
    PROC_TERMINATE_RETRIES = 5
    PROC_TERMINATE_WAIT = 0.1 # seconds

    def __init__(self, exec_env: Environment, ch_env: ChannelFactoryBase,
            no_aslr: bool, startup_input: InputBase, path_retry_limit: int=5):
        # initialize the base class
        super().__init__(exec_env, ch_env, no_aslr)
        self._pobj = None # Popen object of child process
        self._limit = path_retry_limit
        self._startup_input = startup_input

    def __del__(self):
        if self._pobj:
            self._channel.close()

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

    @staticmethod
    def preexec_function():
        os.setpgrp()

    @async_wrapper
    def _launch_target(self):
        # TODO later replace this by a forkserver to reduce reset costs

        ## Kill current process, if any
        if self._pobj:
            return self._channel
            self._channel.close()

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
            restore_signals = False, # TODO check if this should be false
            env = self._exec_env.env,
            preexec_fn = self.preexec_function
        )

        ## Establish a connection
        self._channel = self._ch_env.create(program_name=os.path.basename(self._exec_env.path))

    @property
    def channel(self):
        return self._channel

    def live_path_gen(self, state, sman):
        while True:
            source_state = sman._tracker.current_state
            current_path = []
            if not source_state in sman.state_machine._graph:
                intermediate = min(sman.state_machine._graph.nodes, \
                    key=lambda s: ReachInteraction.l2_distance(
                            (source_state._struct.x, source_state._struct.y),
                            (s._struct.x, s._struct.y))
                        )
                # FIXME the interaction here is not actually used; it is replaced
                # below by the loader
                current_path.append((source_state, intermediate, \
                    ZoomInput((ReachInteraction(source_state, (intermediate._struct.x, intermediate._struct.y)),)))
                )
                source_state = intermediate
            current_path.extend(next(sman.state_machine.get_min_paths(state, source_state)))
            yield current_path

    async def load_state(self, state_or_path: Union[StateBase, list], sman: StateManager, update: bool = True):
        if sman is not None:
            initial_state = sman._tracker.current_state
            await self._channel.clear()
        else:
            await self._launch_target()

        exhaustive = False
        if state_or_path is None or sman is None:
            # special case where the state tracker wants an initial state
            path_gen = ((),)
        elif isinstance((state := state_or_path), StateBase):
            if initial_state == state:
                return
            if not initial_state in sman.state_machine._graph:
                path_gen = None
                exhaustive = True
            else:
                path_gen = (next(sman.state_machine.get_min_paths(state, initial_state)),)#sman.state_machine.get_min_paths(state, initial_state)
        else:
            path_gen = (state_or_path,)
            state = None

        # try out all paths until one succeeds
        paths_tried = 0
        faulty_state = None
        while True:
            if exhaustive and state is not None:
                path_gen = self.live_path_gen(state, sman)
            for path in path_gen:
                if sman is not None and not exhaustive:
                    try:
                        await self.load_state(initial_state, sman, update)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        exhaustive = True
                        continue

                if sman is not None and update:
                    # FIXME should this be done here? (see comment in StateManager.reset_state)
                    sman._last_state = initial_state

                paths_tried += 1
                try:
                    # reconstruct target state by replaying inputs
                    cached_path = list(path)
                    for source, destination, input in cached_path:
                        # check if source matches the current state
                        if source != sman.state_tracker.current_state:
                            faulty_state = source
                            raise StabilityException(
                                f"source state ({source}) did not match current state ({sman.state_tracker.current_state})"
                            )
                        # perform the input
                        dst_x, dst_y, dst_z = map(lambda c: getattr(destination._struct, c), ('x', 'y', 'z'))
                        condition = lambda: sman._tracker.current_state == destination
                        inp = (ReachInteraction(destination, (dst_x, dst_y, dst_z), condition=condition),)
                        inp = ZoomInput(inp)

                        await self.execute_input(inp, sman, update=update)
                        # check if destination matches the current state
                        if destination != sman.state_tracker.current_state:
                            faulty_state = destination
                            raise StabilityException(
                                f"destination state ({destination}) did not match current state ({sman.state_tracker.current_state})"
                            )
                    # at this point, we've succeeded in replaying the path
                    if sman is not None and state is not None:
                        # we only update sman._current_path if it requested a
                        # state; otherwise, it is responsible for saving the
                        # path
                        # FIXME this should all probably be in the loader
                        sman._current_path[:] = list(cached_path)
                    break
                except StabilityException as ex:
                    warning(f"Failed to follow unstable path (reason = {ex.args[0]})! Retrying... ({paths_tried = })")
                    ProfileCount('paths_failed')(1)
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception as ex:
                    # FIXME might want to avoid catching certain exception, e.g. asyncio.CancelledError
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

    async def execute_input(self, input: InputBase, sman: StateManager, update: bool = True):
        if not hasattr(asyncio.get_event_loop(), '_executing_tasks'):
            asyncio.get_event_loop()._executing_tasks = list()

        asyncio.get_event_loop()._executing_tasks.append(asyncio.current_task())
        await super().execute_input(input, sman, update)
        assert asyncio.get_event_loop()._executing_tasks.pop() == asyncio.current_task()
