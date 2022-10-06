from __future__ import annotations
from . import debug, info, warning, critical

from common import StabilityException, StateNotReproducibleException
from typing import Callable
from statemanager import (StateBase,
                         StateMachine,
                         StateTrackerBase,
                         StrategyBase)
from input        import InputBase, MemoryCachingDecorator, FileCachingDecorator
from loader       import StateLoaderBase
from profiler     import ProfileValue, ProfileFrequency, ProfileCount
import asyncio

from interaction import ReachInteraction

class StateManager:
    def __init__(self, loader: StateLoaderBase, tracker: StateTrackerBase,
            strategy_ctor: Callable[[StateMachine, StateBase], StrategyBase],
            workdir: str, protocol: str,
            validate_transitions: bool, minimize_transitions: bool):
        self._loader = loader
        self._tracker = tracker
        self._tracker.state_manager = self
        self._workdir = workdir
        self._protocol = protocol
        self._validate = validate_transitions
        self._minimize = minimize_transitions

        self._last_state = self._tracker.entry_state
        self._last_state.state_manager = self
        self._sm = StateMachine(self._last_state)
        self._strategy = strategy_ctor(self._sm, self._last_state)
        self._current_path = []
        self._reset_current_path()

    @property
    def state_machine(self) -> StateMachine:
        return self._sm

    @property
    def state_tracker(self) -> StateTrackerBase:
        return self._tracker

    def _reset_current_path(self):
        self._current_path.clear()

    @ProfileFrequency('resets')
    async def reset_state(self, state_or_path=None, update=True):
        if update:
            ProfileValue('status')('reset_state')
            self._reset_current_path()
        if self._last_state and update:
            # must clear last state's input whenever a new state is loaded
            # WARN if using SnapshotStateLoader, the snapshot must be taken when
            # the state is first reached, not when a transition is about to
            # happen. Otherwise, last_input may be None, but the snapshot may
            # have residual effect from previous inputs before the state change.
            self._last_state.last_input = None
            # we also set _last_state to the entry state in case the loader
            # needs to execute inputs to reach the target state (e.g. ReplayStateLoader)
            # so that no new edges are added between the last state and the entry state
            # FIXME should this be done by the loader instead?
            # self._last_state = self._tracker.entry_state

        try:
            # FIXME update==True is temporary; should not be considered for main
            # Switch to True to force web UI graph update
            if state_or_path is None:
                current_state = await self._loader.load_state(self._tracker.entry_state, self, update=False)
            else:
                current_state = await self._loader.load_state(state_or_path, self, update=False)
            self._tracker.reset_state(current_state)
            if update:
                if self._tracker.current_state not in self.state_machine._graph:
                    import pdb; pdb.set_trace()
                self._last_state = self._tracker.current_state
                self._strategy.update_state(self._last_state, is_new=False)
        except asyncio.CancelledError:
            # if an interrupt is received while loading a state (e.g. death),
            # self._last_state is not set to the current state because of the
            # exception. Instead, at the next input step, a transition is
            # "discovered" between the last state and the new state, which is
            # wrong
            if update:
                if self._tracker.current_state not in self.state_machine._graph:
                    import pdb; pdb.set_trace()
                self._last_state = self._tracker.current_state
            raise
        except StateNotReproducibleException as ex:
            if update:
                faulty_state = ex._faulty_state
                if faulty_state != self._tracker.entry_state:
                    try:
                        debug(f"Dissolving irreproducible {faulty_state = }")
                        self._sm.dissolve_state(faulty_state, stitch=False)
                        ProfileCount("dissolved_states")(1)
                    except KeyError as ex:
                        warning(f"Faulty state was not even valid")
                    self._strategy.update_state(faulty_state, invalidate=True)
            raise

    async def reload_target(self):
        strategy_target = self._strategy.target
        debug(f"Reloading target state {strategy_target}")
        await self.reset_state(strategy_target)

    def get_context(self, input: InputBase, update: bool) -> StateManagerContext:
        return StateManagerContext(self, input, update)

    async def step(self, input: InputBase):
        """
        Executes the input and updates the state queues according to the
        scheduler. May need to receive information about the current state to
        update it.
        """
        should_reset = self._strategy.step()
        while True:
            try:
                if should_reset:
                    strategy_target = self._strategy.target
                    debug(f'Stepping to new {strategy_target = }')
                    await self.reset_state(strategy_target)
                    target_state = self._tracker.current_state
                    debug(f'Stepped to new {target_state = }')
                break
            except StateNotReproducibleException as ex:
                if isinstance(strategy_target, StateBase):
                    # in case the selected state is unreachable, reset_state()
                    # would have already asked the strategy to invalidate the
                    # faulty state along its path
                    pass
                elif hasattr(strategy_target, '__iter__'):
                    # find and invalidate the transition along the path that
                    # leads to the faulty state
                    try:
                        transition = next(x for x in strategy_target if x[1] == ex._faulty_state)
                        try:
                            debug(f"Deleting irreproducible transition {transition=}")
                            self._sm.delete_transition(transition[0], transition[1])
                        except KeyError as ex:
                            warning(f"Faulty transition was not even valid")
                        self._strategy.update_transition(transition[0], transition[1], transition[2], invalidate=True)
                    except StopIteration:
                        pass
            except Exception as ex:
                # In this case, we need to force the strategy to yield a new
                # target, because we're not entirely sure what went wrong. We
                # invalidate the target state and hope for the best.
                warning(f'Failed to step to new target; invalidating it {ex=}')
                self._strategy.update_state(self._strategy.target_state, invalidate=True)
                raise

        await self._loader.execute_input(input, self, update=True)

    async def update(self, input_gen: Callable[..., InputBase]) -> bool:
        """
        Updates the state machine in case of a state change.

        :param      input_gen:  A function that returns the input that may have
                      resulted in the state change
        :type       input_gen:  Callable[..., InputBase]

        :returns:   Whether or not a new state was reached.
        :rtype:     bool
        """

        updated = False

        # update the current state (e.g., if it needs to track interesting cov)
        self._tracker.update(input_gen)

        # WARN the StateBase object returned by the state tracker may have the
        # same hash(), but may be a different object. This means, any
        # modifications made to the state (or new attributes stored) may not
        # persist.
        current_state = self._tracker.current_state

        # the tracker may return None as current_state, in case it has not yet
        # finished the training phase (preprocessing seeds)
        if current_state is not None:
            new_state, current_state = self._sm.update_state(current_state)
            if new_state:
                current_state.state_manager = self

            self._strategy.update_state(current_state, is_new=new_state)

            debug(f"Updated {'new ' if new_state else ''}{current_state = }")

            if current_state != self._last_state:
                debug(f"Possible transition from {self._last_state} to {current_state}")

                if ReachInteraction.l2_distance(
                    (self._last_state._struct.player_location.x, self._last_state._struct.player_location.y),
                    (current_state._struct.player_location.x, current_state._struct.player_location.y)) > 2000000:
                    # import ipdb; ipdb.set_trace()
                    pass

                if self._last_state.last_input is not None:
                    last_input = self._last_state.last_input + input_gen()
                else:
                    last_input = input_gen()

                stable = True

                if self._validate:
                    try:
                        debug("Attempting to reproduce transition")
                        # we make a copy of _current_path because the loader may
                        # modify it, but at this stage, we may want to restore state
                        # using that path
                        self._last_path = self._current_path.copy()
                        await self.reset_state(self._last_state, update=False)
                        await self._loader.execute_input(last_input, self, update=False)
                        assert current_state == self._tracker.current_state
                        debug(f"{current_state} is reproducible")
                    except AssertionError:
                        # This occurs when the predecessor state was reached through
                        # a different path than that used by reset_state().
                        # The incremental input thus applied to the predecessor
                        # state may have built on top of internal program state that
                        # was not reproduced by reset_state() to arrive at the
                        # successor state.
                        debug(f"Encountered imprecise state ({new_state = })")
                        if new_state:
                            debug(f"Dissolving {current_state = }")
                            self._sm.dissolve_state(current_state)
                            self._strategy.update_state(current_state, invalidate=True)

                            # we save the input for later coverage measurements
                            FileCachingDecorator(self._workdir, "queue", self._protocol)(last_input, self, copy=False, path=self._last_path)
                        stable = False

                        debug(f"Reloading last state ({self._last_state})")
                        self._current_path[:] = self._last_path
                        await self.reset_state(self._current_path, update=False)
                        if self._last_state.last_input is not None:
                            await self._loader.execute_input(self._last_state.last_input, self, update=False)
                        ProfileCount('imprecise')(1)
                        # WARN we set update to be True, despite no change in state,
                        # to force the StateManagerContext object to trim the
                        # input_gen past the troubling interactions, since we won't
                        # be using those interactions in the following iterations.
                        updated = True
                    except (StabilityException, StateNotReproducibleException):
                        # This occurs when the reset_state() encountered an error
                        # trying to reproduce a state, most likely due to an
                        # indeterministic target
                        if new_state:
                            debug(f"Dissolving {current_state = }")
                            self._sm.dissolve_state(current_state)
                            self._strategy.update_state(current_state, invalidate=True)
                        stable = False
                        ProfileCount('unstable')(1)
                    except Exception as ex:
                        debug(f'{ex}')
                        if new_state:
                            debug(f"Dissolving {current_state = }")
                            self._sm.dissolve_state(current_state)
                            self._strategy.update_state(current_state, invalidate=True)
                        stable = False
                        raise

                if stable:
                    last_input = MemoryCachingDecorator()(last_input, copy=False)
                    new_edge = current_state not in self._sm._graph.successors(self._last_state)
                    if (new_state or new_edge) and self._minimize:
                        # call the transition pruning routine to shorten the last input
                        debug("Attempting to minimize transition")
                        try:
                            last_input = await self._minimize_transition(
                                # FIXME self._current_path is not StateBase, as
                                # is required by _minimize_transition, but it
                                # should work, for now
                                self._current_path, current_state, last_input)
                            last_input = FileCachingDecorator(self._workdir, "queue", self._protocol)(last_input, self, copy=False)
                        except Exception as ex:
                            # Minimization failed, again probably due to an
                            # indeterministic target
                            warning(f"Minimization failed, using original input {ex=}")
                            FileCachingDecorator(self._workdir, "queue", self._protocol)(last_input, self, copy=True)

                    self._sm.update_transition(self._last_state, current_state,
                        last_input)
                    self._strategy.update_transition(self._last_state, current_state,
                        last_input)
                    self._current_path.append((self._last_state, current_state, last_input))
                    # reset last state's accumulated input
                    self._last_state.last_input = None
                    self._last_state = current_state
                    info(f'Transitioned to {current_state=}')
                    updated = True

        return updated

    async def _minimize_transition(self, src: StateBase, dst: StateBase, input: InputBase):
        reduced = False

        # Phase 1: perform exponential back-off to find effective tail of input
        ProfileValue('status')('minimize_exp_backoff')
        end = len(input)
        begin = end - 1
        # we do not perform exponential backoff past the midpoint, because that
        # is done in phase 2 anyway
        while begin > end // 2:
            success = True
            await self.reset_state(src, update=False)
            exp_input = input[begin:]
            try:
                await self._loader.execute_input(exp_input, self, update=False)
            except Exception:
                success = False

            success &= dst == self._tracker.current_state
            if success:
                reduced = True
                break
            begin = 0 if (diff := end - (end - begin) * 2) < 0 else diff

        # Phase 2: prune out dead interactions
        ProfileValue('status')('minimize_bin_search')
        if reduced:
            lin_input = input[begin:]
        else:
            lin_input = input
        end = len(input) - begin
        step = (end - 1) // 2
        while step > 0:
            cur = 0
            while cur + step < end:
                success = True
                await self.reset_state(src, update=False)
                tmp_lin_input = lin_input[:cur] + lin_input[cur + step:]
                try:
                    await self._loader.execute_input(tmp_lin_input, self, update=False)
                except Exception:
                    success = False

                success &= dst == self._tracker.current_state
                if success:
                    reduced = True
                    lin_input = MemoryCachingDecorator()(tmp_lin_input, copy=False)
                    end -= step
                else:
                    cur += step
            step //= 2

        # Phase 3: make sure the reduced transition is correct
        await self.reset_state(src, update=False)
        await self._loader.execute_input(lin_input, self, update=False)
        try:
            assert dst == self._tracker.current_state
        except AssertionError:
            raise StabilityException("destination state did not match current state")

        if reduced:
            return lin_input
        else:
            return input

class StateManagerContext:
    """
    This context object provides a wrapper around the input to be sent to the
    target. By wrapping the iterator method, the state manager is updated after
    every interaction in the input. If a state change happens within an input
    sequence, the input is split, where the first part is used as the transition
    and the second part continues to build up state for any following
    transition.
    """
    def __init__(self, sman: StateManager, input: InputBase, update: bool):
        self._sman = sman
        self._input = input
        self._start = self._stop = None
        self._update = update

    def input_gen(self):
        # we delay the call to the slicing decorator until needed
        return self._input[self._start:self._stop]

    async def __aiter__(self):
        self._start = self._stop = 0
        for idx, interaction in enumerate(self._input):
            ProfileValue('status')('fuzz')
            self._stop = idx + 1
            yield interaction
            # the generator execution is suspended until next() is called so
            # the StateManager update is only called after the interaction
            # is executed by the loader
            self._sman._tracker.update(self.input_gen)
            if self._update and await self._sman.update(self.input_gen):
                self._start = idx + 1
            # FIXME The state manager interrupts the target to verify individual
            # transitions. To verify a transition, the last state is loaded, and
            # the last input which observed a new state is replayed. However,
            # due to imprecision of states, the chosen path to the last state
            # may result in a different substate, and the replayed input thus no
            # longer reaches the new state. When such an imprecision is met, the
            # target is kept running, but the path has been "corrupted".
            #
            # It may be better to let the input update the state manager freely
            # all throughout the sequence, and perform the validation step at
            # the end to make sure no interesting states are lost.

        # commit the rest of the input
        if self._sman._last_state.last_input is not None:
            self._sman._last_state.last_input += self.input_gen()
        else:
            self._sman._last_state.last_input = self.input_gen()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
