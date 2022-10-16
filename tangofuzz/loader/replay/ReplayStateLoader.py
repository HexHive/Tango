from .. import warning

from loader       import ProcessLoader
from typing       import Union
from common       import StabilityException, StateNotReproducibleException
from statemanager import StateManager
from tracker import StateBase
from profiler import ProfileCount
from itertools import chain

class ReplayStateLoader(ProcessLoader):
    def __init__(self, /, *, path_retry_limit: int=50, **kwargs):
        super().__init__(**kwargs)
        self._limit = path_retry_limit

    @property
    def channel(self):
        return self._channel

    async def load_state(self, state_or_path: Union[StateBase, list], sman: StateManager) -> StateBase:
        if self.state_tracker is None:
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
                await self._launch_target()

                ## Send startup input
                await self.execute_input(self._generator.startup_input)

                paths_tried += 1
                try:
                    # reconstruct target state by replaying inputs
                    next_state = lambda s, d: self.state_tracker.update(s, d, None, dryrun=True)
                    cached_path = list(path)
                    last_state = None
                    for source, destination, input in cached_path:
                        current_state = next_state(last_state, source)
                        # check if source matches the current state
                        if source != current_state:
                            faulty_state = source
                            raise StabilityException(
                                f"source state ({source}) did not match current state ({current_state})"
                            )
                        # perform the input
                        current_state = next_state(source, destination)
                        await self.execute_input(input)
                        # check if destination matches the current state
                        if destination != current_state:
                            faulty_state = destination
                            raise StabilityException(
                                f"destination state ({destination}) did not match current state ({current_state})"
                            )
                        last_state = current_state
                    # at this point, we've succeeded in replaying the path
                    if sman is not None and state is not None:
                        # we only update sman._current_path if it requested a
                        # state; otherwise, it is responsible for saving the
                        # path
                        # FIXME this should all probably be in the loader
                        sman._current_path[:] = list(cached_path)
                    if self.state_tracker is not None:
                        if not cached_path:
                            destination = self.state_tracker.entry_state
                    else:
                        destination = None
                    return destination
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