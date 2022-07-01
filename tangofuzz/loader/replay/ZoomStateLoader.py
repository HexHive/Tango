from .. import warning

from typing       import Union
from common       import StabilityException, StateNotReproducibleException, sync_to_async
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
from itertools import chain, cycle
from pyroute2 import netns, NetNS
import os
from uuid import uuid4
import asyncio
import networkx as nx
from itertools import islice

class ZoomStateLoader(StateLoaderBase):
    PROC_TERMINATE_RETRIES = 5
    PROC_TERMINATE_WAIT = 0.1 # seconds

    def __init__(self, *args, path_retry_limit: int=5, **kwargs):
        # initialize the base class
        super().__init__(*args, **kwargs)
        self._pobj = None # Popen object of child process
        self._limit = path_retry_limit
        self._loading = False

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

    @sync_to_async()
    def _launch_target(self):
        ## Kill current process, if any
        if self._pobj:
            # return self._channel
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

    def live_path_gen(self, destination_state, sman, source_state=None):
        def edge_weight(src, dst):
            return ReachInteraction.l2_distance(
                (src._struct.player_location.x, src._struct.player_location.y),
                (dst._struct.player_location.x, dst._struct.player_location.y)
            )

        def live_path_gen_helper(source_state, destination_state, visited):
            if destination_state == source_state:
                yield []
                return

            # In case the destination state is not in the graph for some reason,
            # we try to find a path to the nearest in-graph node then bee-line
            # to the destination location.
            tail_path = []
            if destination_state not in sman.state_machine._graph:
                # WARN This can happen due to state imprecision. An interaction
                # can "skip over" a state. However, ZoomStrategy has a bg task
                # that checks current states irrespective of interactions. If
                # it discovers a pickup at a skipped state, it could add it to
                # the stategy's seek list and request a path to it later.
                warning("Path to unknown destination state requested.")
                intermediate = min(
                    filter(lambda s: destination_state._struct.player_location.z - s._struct.player_location.z <= 16,
                        sman.state_machine._graph.nodes), \
                    key=lambda s: edge_weight(destination_state, s))
                # FIXME the input here is not actually used; it is replaced
                # below by the loader calling into the input generator
                tail_path.append((intermediate, destination_state, \
                    ZoomInput()
                ))
                destination_state = intermediate

            head_path = []
            if source_state not in sman.state_machine._graph:
                # WARN This can happen due to state imprecision. An interaction
                # can "skip over" a state. However, ZoomStrategy has a bg task
                # that checks current states irrespective of interactions. If
                # it discovers a pickup at a skipped state, it could add it to
                # the stategy's seek list and request a path to it later.
                warning("Path from unknown source state requested.")
                intermediate = min(
                    filter(lambda s: s._struct.player_location.z - source_state._struct.player_location.z <= 16,
                        sman.state_machine._graph.nodes), \
                    key=lambda s: edge_weight(source_state, s))
                # FIXME the input here is not actually used; it is replaced
                # below by the loader calling into the input generator
                head_path.append((source_state, intermediate, \
                    ZoomInput()
                ))
                source_state = intermediate

            if not nx.has_path(sman.state_machine._graph,
                                source_state, destination_state):
                ancestors = nx.ancestors(sman.state_machine._graph, destination_state)
                ancestors.add(destination_state)
                intermediate = min(ancestors,
                        key=lambda s: edge_weight(source_state, s))
                if intermediate in visited:
                    return
                visited.add(intermediate) # try using destination_state instead
                intermediate_path_gen = sman.state_machine.get_min_paths(destination_state, intermediate)

                if intermediate != source_state:
                    neighbors = sorted(filter(lambda s: s not in ancestors \
                                    and intermediate._struct.player_location.z - s._struct.player_location.z <= 16,
                                           sman.state_machine._graph.nodes),
                            key=lambda s: edge_weight(intermediate, s))[:3] # look in the 3 closest neighbors
                    best_neighbor = None
                    best_path = None
                    for neighbor in neighbors:
                        len_to_neighbor = edge_weight(neighbor, intermediate)
                        if neighbor not in visited:
                            try:
                                for partial_path in islice(live_path_gen_helper(source_state, neighbor, visited), 5): # we try 5 paths at most
                                    path_len = sum(map(lambda p: edge_weight(p[0], p[1]), partial_path)) + len_to_neighbor
                                    if not best_neighbor or path_len < best_len:
                                        best_neighbor = neighbor
                                        best_path = partial_path
                                        best_len = path_len + len_to_neighbor
                            except RecursionError:
                                pass
                    if best_neighbor is not None:
                        best_path.append((best_neighbor, intermediate, ZoomInput()))
                    else:
                        best_path = [(source_state, intermediate, ZoomInput())]
                else:
                    best_path = []

                for path_from_intermediate in intermediate_path_gen:
                    current_path = head_path + best_path + path_from_intermediate + tail_path
                    yield current_path
            else:
                direct_path_gen = sman.state_machine.get_min_paths(destination_state, source_state)
                for path_from_source in direct_path_gen:
                    current_path = head_path + path_from_source + tail_path
                    yield current_path

        def live_path_gen_corrector(source_state, destination_state, visited):
            for path in live_path_gen_helper(source_state, destination_state, visited):
                if not path:
                    yield next(sman.state_machine.get_min_paths(source_state, source_state))
                else:
                    yield path

        source_state = source_state or sman._tracker.current_state
        return cycle(live_path_gen_corrector(source_state, destination_state, set()))

    async def load_state(self, state_or_path: Union[StateBase, list], sman: StateManager, update: bool = True):
        already_loading = self._loading
        try:
            self._loading = True
            if sman is not None:
                initial_state = sman._tracker.current_state
                # whenever a new state is requested, we stop sending inputs
                await self._channel.clear()
                # if initial_state not in sman.state_machine._graph:
                #     # We have somehow arrived at a state that has not been seen
                #     # previously. The only clean way to recover is to relaunch
                #     # the target to restore the entry state.
                #     # FIXME gotta inform the strategy somehow that the target
                #     # has been reset
                #     await self._launch_target()
                #     sman._strategy._seek.clear()
                #     # while sman._tracker.current_state != sman._tracker.entry_state:
                #         # await DelayInteraction(0.01).perform(self._channel)
                #     initial_state = sman._tracker.current_state
                #     assert initial_state == sman._tracker.entry_state

                #     # FIXME At this point, if state_or_path is a path, it might
                #     # have assumed a different initial state for the first
                #     # transition. If that's the case, we try to reconstruct it.
                #     if not isinstance(state_or_path, StateBase) and state_or_path is not None:
                #         path_gen = self.live_path_gen(state_or_path[-1][1], sman)
                #         state_or_path = None
                #         state = None
                if update:
                    sman._last_state = initial_state
            else:
                await self._launch_target()

            exhaustive = False
            if state_or_path is None or sman is None:
                # special case where the state tracker wants an initial state
                path_gen = ((),)
            elif isinstance((state := state_or_path), StateBase):
                if initial_state == state:
                    return
                # if not initial_state in sman.state_machine._graph:
                # the exhaustive search accounts for initial states that are not in
                # the graph by bee-lining to the closest in-graph state
                path_gen = None
                exhaustive = True
                # else:
                    # path_gen = (next(sman.state_machine.get_min_paths(state, initial_state)),)#sman.state_machine.get_min_paths(state, initial_state)
            elif state_or_path is not None:
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
                            # see comment in sman.reset_state
                            if sman._tracker.current_state not in sman.state_machine._graph:
                                import pdb; pdb.set_trace()
                            sman._last_state = sman._tracker.current_state
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
                        follow_inp = self._generator.generate_follow_path(cached_path)
                        for (source, destination, _), interaction in zip(cached_path, follow_inp):
                            # src_x, src_y, src_z = map(lambda c: getattr(source._struct.player_location, c), ('x', 'y', 'z'))
                            # dst_x, dst_y, dst_z = map(lambda c: getattr(destination._struct.player_location, c), ('x', 'y', 'z'))
                            # if dst_z - src_z > 16:
                            #     # it is not possible to go up more than 16 units in one move
                            #     faulty_state = destination
                            #     # We use this to force yield because so far we have
                            #     # not. If the source state is in fact not in the
                            #     # graph, the state manager will request a new target
                            #     # from the strategy and will try to reload again,
                            #     # but on no point along that path do we yield, so
                            #     # we starve other tasks from running. In one case,
                            #     # the strategy feedback task could be essential to
                            #     # get us out of this sticky situation.
                            #     # await asyncio.sleep(0)
                            #     await self._launch_target()
                            #     raise StateNotReproducibleException("target state too high", destination)

                            # check if source matches the current state
                            if source != (current_state := sman.state_tracker.current_state):
                                faulty_state = source
                                raise StabilityException(
                                    f"source state ({source}) did not match current state ({current_state})"
                                )
                            # at this point, the faulty state can only be the
                            # destination
                            faulty_state = destination
                            # perform the follow interaction
                            inp = ZoomInput((interaction,))
                            await self.execute_input(inp, sman, update=update)
                            # check if destination matches the current state
                            if destination != (current_state := sman.state_tracker.current_state):
                                raise StabilityException(
                                    f"destination state ({destination}) did not match current state ({current_state})"
                                )
                        # at this point, we've succeeded in replaying the path
                        if sman is not None and state is not None:
                            # we only update sman._current_path if it requested a
                            # state; otherwise, it is responsible for saving the
                            # path
                            # FIXME this should all probably be in the loader
                            sman._current_path[:] = cached_path
                        break
                    except StabilityException as ex:
                        warning(f"Failed to follow unstable path (reason = {ex.args[0]})! Retrying... ({paths_tried = })")
                        ProfileCount('paths_failed')(1)
                        continue
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
        finally:
            if not already_loading:
                self._loading = False


    async def execute_input(self, input: InputBase, sman: StateManager, update: bool = True):
        if not hasattr(asyncio.get_running_loop(), '_executing_tasks'):
            asyncio.get_running_loop()._executing_tasks = list()

        asyncio.get_running_loop()._executing_tasks.append(asyncio.current_task())
        await super().execute_input(input, sman, update)
        assert asyncio.get_running_loop()._executing_tasks.pop() == asyncio.current_task()
