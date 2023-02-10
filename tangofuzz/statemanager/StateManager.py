from __future__ import annotations
from . import debug, info, warning, critical

from common import (StabilityException, StateNotReproducibleException,
                    StatePrecisionException, CoroInterrupt)
from typing import Callable
from statemanager import StateMachine
from statemanager.strategy import StrategyBase
from tracker import StateBase, StateTrackerBase
from input        import InputBase, DecoratorBase, MemoryCachingDecorator, FileCachingDecorator
from generator    import InputGeneratorBase
from loader       import StateLoaderBase # FIXME there seems to be a cyclic dep
from profiler     import ProfileValue, ProfileFrequency, ProfileCount, ProfileLambda
import asyncio
from common import async_enumerate

class StateManager:
    def __init__(self, generator: InputGeneratorBase, loader: StateLoaderBase,
            tracker: StateTrackerBase,
            strategy_ctor: Callable[[StateMachine, StateBase], StrategyBase],
            workdir: str, protocol: str,
            validate_transitions: bool, minimize_transitions: bool):
        self._generator = generator
        self._loader = loader
        self._tracker = tracker
        self._tracker.state_manager = self
        self._workdir = workdir
        self._protocol = protocol
        self._validate = validate_transitions
        self._minimize = minimize_transitions

        self._last_state = self._current_state = self.state_tracker.entry_state
        self._last_state.state_manager = self
        self._sm = StateMachine(self._last_state)
        self._strategy = strategy_ctor(self._sm, self._last_state)
        self._current_path = []

        ProfileLambda('global_cov')(lambda: sum(map(lambda x: x._set_count + x._clr_count, filter(lambda x: x != self.state_tracker.entry_state, self._sm._graph.nodes))))

    @property
    def state_machine(self) -> StateMachine:
        return self._sm

    @property
    def state_tracker(self) -> StateTrackerBase:
        return self._tracker

    def _reset_current_path(self):
        self._current_path.clear()

    @ProfileFrequency('resets')
    async def reset_state(self, state_or_path=None, dryrun=False) -> StateBase:
        if not dryrun:
            ProfileValue('status')('reset_state')
            self._reset_current_path()
            # must clear last state's input whenever a new state is
            # loaded
            #
            # WARN if using SnapshotStateLoader, the snapshot must be
            # taken when the state is first reached, not when a
            # transition is about to happen. Otherwise, last_input may
            # be None, but the snapshot may have residual effect from
            # previous inputs before the state change.
            if self._last_state is not None:
                self._last_state.last_input = None

        try:
            if state_or_path is None:
                current_state = await self._loader.load_state(self.state_tracker.entry_state, self)
            else:
                current_state = await self._loader.load_state(state_or_path, self)
            if not dryrun:
                self.state_tracker.reset_state(current_state)
                if (current_state := self.state_tracker.current_state) not in self.state_machine._graph:
                    critical("Current state is not in state graph! Launching interactive debugger...")
                    import ipdb; ipdb.set_trace()
                self._last_state = current_state
                self._strategy.update_state(self._last_state, input=None)
        except CoroInterrupt:
            # if an interrupt is received while loading a state (e.g. death),
            # self._last_state is not set to the current state because of the
            # exception. Instead, at the next input step, a transition is
            # "discovered" between the last state and the new state, which is
            # wrong
            if not dryrun:
                if (current_state := self.state_tracker.current_state) not in self.state_machine._graph:
                    critical("Current state is not in state graph! Launching interactive debugger...")
                    import ipdb; ipdb.set_trace()
                self._last_state = current_state
            raise
        except StateNotReproducibleException as ex:
            if not dryrun:
                faulty_state = ex._faulty_state
                if faulty_state != self.state_tracker.entry_state:
                    try:
                        debug(f"Dissolving irreproducible {faulty_state = }")
                        self._sm.dissolve_state(faulty_state, stitch=False)
                        ProfileCount("dissolved_states")(1)
                    except KeyError as ex:
                        warning(f"Faulty state was not even valid")
                    self._strategy.update_state(faulty_state, input=None, exc=ex)
            raise
        return current_state

    async def reload_target(self):
        strategy_target = self._strategy.target
        debug(f"Reloading target state {strategy_target}")
        await self.reset_state(strategy_target)

    def get_context_input(self, input: InputBase) -> StateManagerContext:
        return StateManagerContext(self)(input)

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
                    await self.reload_target()
                    debug(f'Stepped to new {self._strategy.target = }')
                break
            except StateNotReproducibleException as ex:
                strategy_target = self._strategy.target
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
                        self._strategy.update_transition(*transition, state_changed=transition[0]==transition[1],
                            exc=ex)
                    except StopIteration:
                        pass
            except Exception as ex:
                # In this case, we need to force the strategy to yield a new
                # target, because we're not entirely sure what went wrong. We
                # invalidate the target state and hope for the best.
                warning(f'Failed to step to new target; invalidating it {ex=}')
                self._strategy.update_state(self._strategy.target_state, input=None, exc=ex)
                raise

        context_input = self.get_context_input(input)
        await self._loader.execute_input(context_input)

    async def update(self, input_gen: Callable[..., InputBase]) -> tuple(bool, InputBase):
        """
        Updates the state machine in case of a state change.

        :param      input_gen:  A function that returns the input that may have
                      resulted in the state change
        :type       input_gen:  Callable[..., InputBase]

        :returns:   (state_changed?, accumulated input)
        :rtype:     tuple(bool, InputBase)
        """

        updated = False
        stable = True

        # WARN the StateBase object returned by the state tracker may have the
        # same hash(), but may be a different object. This means, any
        # modifications made to the state (or new attributes stored) may not
        # persist.
        current_state = self.state_tracker.current_state
        last_input = None

        # the tracker may return None as current_state, in case it has not yet
        # finished the training phase (preprocessing seeds)
        if current_state is not None:
            # we obtain a persistent reference to the current_state
            is_new_state, self._current_state = self._sm.update_state(current_state)
            if is_new_state:
                self._current_state.state_manager = self

            debug(f"Updated {'new ' if is_new_state else ''}{self._current_state = }")

            if self._current_state != self._last_state:
                debug(f"Possible transition from {self._last_state} to {self._current_state}")

                input = input_gen()
                if self._validate:
                    try:
                        debug("Attempting to reproduce transition")
                        # we make a copy of _current_path because the loader may
                        # modify it, but at this stage, we may want to restore state
                        # using that path
                        self._last_path = self._current_path.copy()
                        await self.reset_state(self._last_state, dryrun=True)
                        await self._loader.execute_input(input)
                        assert self._current_state == self.state_tracker.peek(self._last_state, self._current_state)
                        debug(f"{self._current_state} is reproducible")
                    except AssertionError as ex:
                        # This occurs when the predecessor state was reached through
                        # a different path than that used by reset_state().
                        # The incremental input thus applied to the predecessor
                        # state may have built on top of internal program state that
                        # was not reproduced by reset_state() to arrive at the
                        # successor state.
                        debug(f"Encountered imprecise state ({is_new_state = })")
                        if is_new_state:
                            debug(f"Dissolving {self._current_state = }")
                            self._sm.dissolve_state(self._current_state)

                            # we save the input for later coverage measurements
                            FileCachingDecorator(self._workdir, "unstable", self._protocol)(input, self, copy=False, path=self._last_path)
                        stable = False
                        raise StatePrecisionException(f"{self._current_state} was reached through an imprecise path") from ex
                    except (StabilityException, StateNotReproducibleException):
                        # This occurs when the reset_state() encountered an error
                        # trying to reproduce a state, most likely due to an
                        # indeterministic target
                        if is_new_state:
                            debug(f"Dissolving {self._current_state = }")
                            self._sm.dissolve_state(self._current_state)
                        stable = False
                        raise
                    except Exception as ex:
                        debug(f'{ex}')
                        if is_new_state:
                            debug(f"Dissolving {self._current_state = }")
                            self._sm.dissolve_state(self._current_state)
                        stable = False
                        raise

                if stable:
                    last_input = MemoryCachingDecorator()(input, copy=False)
                    is_new_edge = self._current_state not in self._sm._graph.successors(self._last_state)
                    if (is_new_state or is_new_edge) and self._minimize:
                        # call the transition pruning routine to shorten the last input
                        debug("Attempting to minimize transition")
                        try:
                            last_input = await self._minimize_transition(
                                self._current_path, self._current_state, last_input)
                            last_input = FileCachingDecorator(self._workdir, "queue", self._protocol)(last_input, self, copy=False)
                        except Exception as ex:
                            # Minimization failed, again probably due to an
                            # indeterministic target
                            warning(f"Minimization failed, saving original input {ex=}")
                            FileCachingDecorator(self._workdir, "unstable", self._protocol)(last_input, self, copy=True)
                            stable = False
                            raise

                    # FIXME doesn't work as intended, but could save from recalculating cov map diffs if it works
                    # if self._current_state != self.state_tracker.peek(self._last_state):
                    #     raise StabilityException("Failed to obtain consistent behavior")
                    # self.state_tracker.update(self._last_state, last_input, peek_result=self._current_state)
                    if self._current_state != self.state_tracker.update(self._last_state, last_input):
                        raise StabilityException("Failed to obtain consistent behavior")

                    self._sm.update_transition(self._last_state, self._current_state,
                        last_input)
                    self._last_state = self._current_state
                    info(f'Transitioned to {self._current_state=}')
                    updated = True

        return updated, last_input

    async def _minimize_transition(self, src: Union[list, StateBase], dst: StateBase, input: InputBase):
        if isinstance(src, list):
            src = src[-1][1]

        reduced = False

        # Phase 1: perform exponential back-off to find effective tail of input
        ProfileValue('status')('minimize_exp_backoff')
        end = len(input)
        begin = end - 1
        # we do not perform exponential backoff past the midpoint, because that
        # is done in phase 2 anyway
        while begin > end // 2:
            success = True
            await self.reset_state(src, dryrun=True)
            exp_input = input[begin:]
            try:
                await self._loader.execute_input(exp_input)
            except Exception:
                success = False

            success &= dst == self.state_tracker.peek(src, dst)
            if success:
                reduced = True
                break
            begin = 0 if (diff := end - (end - begin) * 2) < 0 else diff

        # Phase 2: prune out dead interactions
        if reduced:
            lin_input = input[begin:]
        else:
            lin_input = input
        end = len(input) - begin
        step = (end - 1) // 2
        i = 0
        while step > 0:
            cur = 0
            while cur + step < end:
                ProfileValue('status')(f'minimize_bin_search ({100*i/(2 * (len(input) - begin)):.1f}%)')
                success = True
                await self.reset_state(src, dryrun=True)
                tmp_lin_input = lin_input[:cur] + lin_input[cur + step:]
                try:
                    await self._loader.execute_input(tmp_lin_input)
                except Exception:
                    success = False

                success &= dst == self.state_tracker.peek(src, dst)
                if success:
                    reduced = True
                    lin_input = MemoryCachingDecorator()(tmp_lin_input, copy=False)
                    end -= step
                else:
                    cur += step
                i += 1
            step //= 2

        # Phase 3: make sure the reduced transition is correct
        await self.reset_state(src, dryrun=True)
        await self._loader.execute_input(lin_input)
        try:
            assert dst == self.state_tracker.peek(src, dst)
        except AssertionError as ex:
            raise StabilityException("destination state did not match current state") from ex

        if reduced:
            return lin_input
        else:
            return input

class StateManagerContext(DecoratorBase):
    """
    This context object provides a wrapper around the input to be sent to the
    target. By wrapping the iterator method, the state manager is updated after
    every interaction in the input. If a state change happens within an input
    sequence, the input is split, where the first part is used as the transition
    and the second part continues to build up state for any following
    transition.
    """
    def __init__(self, sman: StateManager):
        self._sman = sman
        self._start = self._stop = None

    def input_gen(self):
        # we delay the call to the slicing decorator until needed
        head = self._sman._last_state.last_input
        tail = self._input[self._start:self._stop]
        if head is None:
            if self._start >= self._stop:
                return None
            else:
                return tail
        else:
            return head + tail

    def update_state(self, *args, **kwargs):
        self._sman._strategy.update_state(*args, **kwargs)
        self._sman._generator.update_state(*args, **kwargs)

    def update_transition(self, *args, **kwargs):
        self._sman._strategy.update_transition(*args, **kwargs)
        self._sman._generator.update_transition(*args, **kwargs)

    @property
    def orig_input(self):
        # we pop the StateManagerContext decorator itself to obtain a reference
        # to the original input, typically the output of the input generator
        return self._input.pop_decorator()[0]

    async def ___aiter___(self, input, orig):
        self._start = self._stop = 0
        idx = -1
        async for idx, interaction in async_enumerate(orig()):
            ProfileValue('status')('fuzz')
            self._stop = idx + 1
            yield interaction
            # the generator execution is suspended until next() is called so
            # the StateManager update is only called after the interaction
            # is executed by the loader

            updated = False
            last_input = None
            try:
                last_state = self._sman._last_state
                updated, last_input = await self._sman.update(self.input_gen)
                if updated:
                    self._start = idx + 1

                    # if an update has happened, we've transitioned out of the
                    # last_state and as such, it's no longer necessary to keep
                    # track of the last input
                    last_state.last_input = None
            except Exception as ex:
                self._stop = idx + 1
                # we also clear the last input on an exception, because the next
                # course of action will involve a reset_state
                last_state.last_input = None
                self.update_state(self._sman._current_state,
                    input=self.input_gen(), orig_input=self.orig_input, exc=ex)
                raise
            else:
                if not updated:
                    # we force a state tracker update (e.g. update implicit state)
                    # FIXME this may not work well with all state trackers
                    ns = self._sman.state_tracker.update(last_state, last_input)
                    assert ns == last_state, "State tracker is inconsistent!"
                else:
                    self._sman._current_path.append(
                        (last_state, self._sman._current_state, last_input))

                self.update_state(self._sman._current_state, input=last_input,
                    orig_input=self.orig_input)
                self.update_transition(
                    last_state, self._sman._current_state,
                    input=last_input, orig_input=self.orig_input,
                    state_changed=updated)

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

        if idx >= 0:
            # commit the rest of the input
            self._sman._last_state.last_input = self.input_gen()

    def ___iter___(self, input, orig):
        return orig()
