from __future__ import annotations
from . import debug, info, critical

from common import StabilityException, StatePrecisionException, StateNotReproducibleException
from typing import Callable
from statemanager import (StateBase,
                         StateMachine,
                         StateTrackerBase,
                         StrategyBase)
from input        import InputBase, MemoryCachingDecorator, FileCachingDecorator
from loader       import StateLoaderBase
from profiler     import ProfileValue, ProfileFrequency, ProfileCount

class StateManager:
    class StateManagerContext:
    def __init__(self, loader: StateLoaderBase, tracker: StateTrackerBase,
        def __init__(self, sman: StateManager, input: InputBase):
            strategy_ctor: Callable[[StateMachine, StateBase], StrategyBase],
            self._sman = sman
            self._input = input
            self._start = self._stop = None

        def input_gen(self):
            # we delay the call to the slicing decorator until needed
            return self._input[self._start:self._stop]

        def __iter__(self):
            self._start = self._stop = 0
            for idx, interaction in enumerate(self._input):
                ProfileValue('status')('fuzz')
                self._stop = idx + 1
                yield interaction
                # the generator execution is suspended until next() is called so
                # the StateManager update is only called after the interaction
                # is executed by the loader
                if self._sman.update(self.input_gen):
                    self._start = idx + 1

            # commit the rest of the input
            if self._sman._last_state.last_input is not None:
                self._sman._last_state.last_input += self.input_gen()
            else:
                self._sman._last_state.last_input = self.input_gen()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            pass

            cache_inputs: bool, workdir: str, protocol: str):
        self._loader = loader
        self._tracker = tracker
        self._cache_inputs = cache_inputs
        self._workdir = workdir
        self._protocol = protocol

        self._last_state = self._tracker.entry_state
        self._sm = StateMachine(self._last_state)
        self._strategy = strategy_ctor(self._sm, self._last_state)

    @property
    def state_machine(self) -> StateMachine:
        return self._sm

    @property
    def state_tracker(self) -> StateTrackerBase:
        return self._tracker

    @ProfileFrequency('resets')
    def reset_state(self, state_or_path=None, update=True):
        if update:
            ProfileValue('status')('reset_state')
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
            self._last_state = self._tracker.entry_state

        try:
            if state_or_path is None:
                self._loader.load_state(self._tracker.entry_state, self, update=False)
            else:
                self._loader.load_state(state_or_path, self, update=False)
            if update:
                self._last_state = self._tracker.current_state
        except StateNotReproducibleException as ex:
            if update:
                faulty_state = ex._faulty_state
                debug(f"Dissolving irreproducible {faulty_state = }")
                self._sm.dissolve_state(faulty_state)
                self._strategy.update_state(faulty_state, invalidate=True)
                ProfileCount("dissolved_states")(1)
            raise

    def reload_target(self):
        self.reset_state(self._strategy.target)

    def get_context(self, input: InputBase) -> StateManager.StateManagerContext:
        return self.StateManagerContext(self, input)

    def step(self, input: InputBase):
        """
        Executes the input and updates the state queues according to the
        scheduler. May need to receive information about the current state to
        update it.
        """
        while True:
            try:
                should_reset = self._strategy.step()
                if should_reset:
                    self.reset_state(self._strategy.target)
                    target_state = self._tracker.current_state
                    debug(f'Stepped to new {target_state = }')
                break
            except StateNotReproducibleException as ex:
                if isinstance(target, StateBase):
                    # in case the selected state is unreachable, reset_state()
                    # would have already asked the strategy to invalidate the
                    # faulty state along its path
                    pass
                elif hasattr(target, '__iter__'):
                    # find and invalidate the transition along the path that
                    # leads to the faulty state
                    transition = next(x for x in target if x[1] == ex._faulty_state)
                    self._strategy.update_transition(transition[0], transition[1], invalidate=True)
            except Exception:
                # In this case, we need to force the strategy to yield a new
                # target, because we're not entirely sure what went wrong. We
                # invalidate the target state and hope for the best.
                self._strategy.update_state(self._strategy.target_state, invalidate=True)

        self._loader.execute_input(input, self, update=True)

    def update(self, input_gen: Callable[..., InputBase]) -> bool:
        """
        Updates the state machine in case of a state change.

        :param      input_gen:  A function that returns the input that may have
                      resulted in the state change
        :type       input_gen:  Callable[..., InputBase]

        :returns:   Whether or not a new state was reached.
        :rtype:     bool
        """

        updated = False
        # WARN the StateBase object returned by the state tracker may have the
        # same hash(), but may be a different object. This means, any
        # modifications made to the state (or new attributes stored) may not
        # persist.
        current_state = self._tracker.current_state

        # update the current state (e.g., if it needs to track interesting cov)
        self._tracker.update(self._last_state, current_state, input_gen)

        # the tracker may return None as current_state, in case it has not yet
        # finished the training phase (preprocessing seeds)
        if current_state is not None:
            new = self._sm.update_state(current_state)
            self._strategy.update_state(current_state, is_new=new)
            debug(f"Updated {'new ' if new else ''}{current_state = }")
            if current_state != self._last_state:
                debug(f"Possible transition from {self._last_state} to {current_state}")
                if self._last_state.last_input is not None:
                    last_input = self._last_state.last_input + input_gen()
                else:
                    last_input = input_gen()

                stable = True

                try:
                    debug("Attempting to reproduce transition")
                    self.reset_state(self._last_state, update=False)
                    self._loader.execute_input(last_input, self, update=False)
                    assert current_state == self._tracker.current_state
                    debug(f"{current_state} is reproducible")
                except AssertionError:
                    # This occurs when the predecessor state was reached through
                    # a different path than that used by reset_state().
                    # The incremental input thus applied to the predecessor
                    # state may have built on top of internal program state that
                    # was not reproduced by reset_state() to arrive at the
                    # successor state.
                    debug(f"Encountered imprecise state ({new = })")
                    if new:
                        debug(f"Dissolving {current_state = }")
                        self._sm.dissolve_state(current_state)
                        self._strategy.update_state(current_state, invalidate=True)
                    stable = False

                    ProfileCount('imprecise')(1)
                except (StabilityException, StateNotReproducibleException):
                    # This occurs when the reset_state() encountered an error
                    # trying to reproduce a state, most likely due to an
                    # indeterministic target
                    if new:
                        debug(f"Dissolving {current_state = }")
                        self._sm.dissolve_state(current_state)
                        self._strategy.update_state(current_state, invalidate=True)
                    stable = False
                    ProfileCount('unstable')(1)
                except Exception as ex:
                    critical(f'{ex}')
                    if new:
                        debug(f"Dissolving {current_state = }")
                        self._sm.dissolve_state(current_state)
                        self._strategy.update_state(current_state, invalidate=True)
                    stable = False
                    raise

                if stable:
                    if self._cache_inputs:
                        last_input = MemoryCachingDecorator()(last_input, copy=False)
                        if new:
                            # call the transition pruning routine to shorten the last input
                            debug("Attempting to minimize transition")
                            try:
                                last_input = self._minimize_transition(
                                    self._last_state, current_state, last_input)
                            except Exception as ex:
                                # Minimization failed, again probably due to an
                                # indeterministic target
                                debug(f"Minimization failed, using original input {ex=}")
                                FileCachingDecorator(self._workdir, "queue", self._protocol)(last_input, self, copy=True)

                    self._sm.update_transition(self._last_state, current_state,
                        last_input)
                    self._strategy.update_transition(self._last_state, current_state)
                    self._last_state = current_state
                    info(f'Transitioned to {current_state=}')
                    updated = True

        return updated

    def _minimize_transition(self, src: StateBase, dst: StateBase, input: InputBase):
        reduced = False

        # Phase 1: perform exponential back-off to find effective tail of input
        ProfileValue('status')('minimize_exp_backoff')
        end = len(input)
        begin = end - 1
        # we do not perform exponential backoff past the midpoint, because that
        # is done in phase 2 anyway
        while begin > end // 2:
            success = True
            self.reset_state(src, update=False)
            exp_input = input[begin:]
            try:
                self._loader.execute_input(exp_input, self, update=False)
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
                self.reset_state(src, update=False)
                tmp_lin_input = lin_input[:cur] + lin_input[cur + step:]
                try:
                    self._loader.execute_input(tmp_lin_input, self, update=False)
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
        self.reset_state(src, update=False)
        self._loader.execute_input(lin_input, self, update=False)
        try:
            assert dst == self._tracker.current_state
        except AssertionError:
            raise StabilityException("destination state did not match current state")

        if reduced:
            result = FileCachingDecorator(self._workdir, "queue", self._protocol)(lin_input, self, copy=False)
        else:
            result = FileCachingDecorator(self._workdir, "queue", self._protocol)(input, self, copy=True)

        return result
