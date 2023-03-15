from __future__ import annotations
from . import debug, info, warning, critical

from tango.exceptions import (StabilityException, StateNotReproducibleException,
    StatePrecisionException)
from tango.core.tracker import AbstractState, AbstractTracker
from tango.core.input import (AbstractInput, BaseInput, PreparedInput,
    BaseDecorator)
from tango.core.driver import AbstractDriver
from tango.core.loader import AbstractLoader
from tango.core.profiler import (ValueProfiler, FrequencyProfiler,
    CountProfiler)
from tango.core.types import (Path, LoadableTarget,
    ExplorerStateUpdateCallback, ExplorerTransitionUpdateCallback,
    ExplorerStateReloadCallback)
from tango.common import AsyncComponent, ComponentType, ComponentOwner

from abc import ABC, abstractmethod
from typing import Callable
from itertools import chain

__all__ = [
    'AbstractExplorer', 'BaseExplorer'
]

class AbstractExplorer(AsyncComponent, ABC,
        component_type=ComponentType.explorer):
    def __init__(self):
        self._state_reload_cb = self._nop_cb
        self._state_update_cb = self._nop_cb
        self._transition_update_cb = self._nop_cb

    @staticmethod
    async def _nop_cb(*args, **kwargs):
        pass

    def register_state_reload_callback(self,
            callback: ExplorerStateReloadCallback):
        self._state_reload_cb = callback

    def register_state_update_callback(self,
            callback: ExplorerStateUpdateCallback):
        self._state_update_cb = callback

    def register_transition_update_callback(self,
            callback: ExplorerTransitionUpdateCallback):
        self._transition_update_cb = callback

    @abstractmethod
    async def reload_state(self, state_or_path=None) -> AbstractState:
        pass

    @abstractmethod
    async def follow(self, input: AbstractInput):
        pass

class BaseExplorer(AbstractExplorer,
        capture_components={
            ComponentType.loader, ComponentType.tracker, ComponentType.driver},
        capture_paths=['explorer.reload_attempts']):
    def __init__(self, *,
            loader: AbstractLoader, tracker: AbstractTracker,
            driver: AbstractDriver, reload_attempts: str='50', **kwargs):
        super().__init__(**kwargs)
        self._loader = loader
        self._tracker = tracker
        self._driver = driver
        self._reload_attempts = int(reload_attempts)
        self._current_path = []
        self._last_path = []

    async def finalize(self, owner: ComponentOwner):
        self._last_state = self._current_state = self._tracker.entry_state
        await super().finalize(owner)

    @property
    def tracker(self) -> AbstractTracker:
        return self._tracker

    @property
    def loader(self) -> AbstractLoader:
        return self._loader

    @property
    def driver(self) -> AbstractDriver:
        return self._driver

    def _reset_current_path(self):
        self._current_path.clear()

    async def attempt_load_state(self, loadable: LoadableTarget):
        if isinstance(state := loadable, AbstractState):
            # FIXME fetch the paths via the tracker; a non-replay loader would
            # be able to reproduce the final state without loading intermediate
            # states. The tracker would then return a single-transition "path".
            paths = chain(self._tracker.state_graph.get_min_paths(state),
                          self._tracker.state_graph.get_paths(state))
            # Loop over possible paths until retry threshold
            for i, path in zip(range(self._reload_attempts), paths):
                try:
                    return await self._arbitrate_load_state(path)
                except StabilityException as ex:
                    warning("Failed to follow unstable path"
                        f" (reason = {ex.args[0]})!"
                        f" Retrying... ({i+1}/{self._reload_attempts})")
                    CountProfiler('unstable')(1)
            raise StateNotReproducibleException("destination state not reproducible",
                state)
        else:
            try:
                return await self._arbitrate_load_state(loadable)
            except StabilityException as ex:
                warning("Failed to follow unstable preselected path"
                    f" (reason = {ex.args[0]})!")
                CountProfiler('unstable')(1)
                raise StateNotReproducibleException("destination state not reproducible",
                    None) from ex

    async def _arbitrate_load_state(self, path: Path):
        # FIXME interactions between the loader, tracker, explorer, and possibly
        # higher components (generator, mutator, fuzzer, etc...) could be more
        # conventiently specified by an arbitration protocol which all concerned
        # components implement.

        # clear the current path to stay consistent with the target
        self._reset_current_path()
        # deliberately not `await`ed; we exchange messages with the generator
        load = self._loader.load_state(path)
        try:
            nxt = None
            while True:
                previous, current, inp = await load.asend(nxt)
                if current is None:
                    raise RuntimeError("This is deprecated behavior.")
                else:
                    if inp:
                        self._current_path.append((previous, current, inp))
                    nxt = self._tracker.peek(previous, current)
        except StopAsyncIteration:
            return current

    @FrequencyProfiler('resets')
    async def reload_state(self, state_or_path: LoadableTarget=None, *,
            dryrun=False) -> AbstractState:
        if not dryrun:
            ValueProfiler('status')('reset_state')
            self._reset_current_path()
            # must clear last state's input whenever a new state is
            # loaded
            #
            # WARN if using SnapshotLoader, the snapshot must be
            # taken when the state is first reached, not when a
            # transition is about to happen. Otherwise, last_input may
            # be None, but the snapshot may have residual effect from
            # previous inputs before the state change.
            if self._last_state is not None:
                self._last_state.last_input = None

        loadable = state_or_path or self._tracker.entry_state
        try:
            current_state = await self.attempt_load_state(loadable)
            if not dryrun:
                self._tracker.reset_state(current_state)
                self._last_state = current_state
        except StateNotReproducibleException as ex:
            if not dryrun:
                self._tracker.update_state(ex.faulty_state, input=None, exc=ex)
            await self._state_reload_cb(loadable, exc=ex)
            raise
        return current_state

    def get_context_input(self, input: BaseInput, **kwargs) -> BaseExplorerContext:
        return BaseExplorerContext(self, **kwargs)(input)

    async def follow(self, input: BaseInput, **kwargs):
        """
        Executes the input and updates the state queues according to the
        scheduler. May need to receive information about the current state to
        update it.
        """
        context_input = self.get_context_input(input, **kwargs)
        await self._driver.execute_input(context_input)

    async def update(self, input_gen: Callable[..., BaseInput],
            minimize: bool=True, validate: bool=True) \
            -> tuple(bool, bool, BaseInput):
        """
        Updates the state machine in case of a state change.

        :param      input_gen:  A function that returns the input that may have
                      resulted in the state change
        :type       input_gen:  Callable[..., BaseInput]

        :returns:   (state_changed?, unseen?, accumulated input)
        :rtype:     tuple(bool, bool, BaseInput)
        """

        # WARN the AbstractState object returned by the state tracker may have the
        # same hash(), but may be a different object. This means, any
        # modifications made to the state (or new attributes stored) may not
        # persist.
        current_state = self._tracker.current_state

        # the tracker may return None as current_state, in case it has not yet
        # finished the training phase (preprocessing seeds)
        if current_state is None:
            return False, False, None

        self._current_state = current_state

        debug(f"Reached {current_state = }")

        if self._current_state == self._last_state:
            return False, False, None

        unseen = self._current_state not in self._tracker.state_graph.successors(self._last_state)
        debug(f"Possible transition from {self._last_state} to {self._current_state}")

        last_input = input_gen()
        if unseen:
            if minimize:
                # During minimization, the input is sliced/joined/cached, which
                # are facilities provided by BaseInput.
                # An input type that does not support these capabilities must
                # inherit from AbstractInput and disable minimization (or use a
                # different Explorer).
                assert isinstance(last_input, BaseInput), \
                    "Minimizing requires inputs to be BaseInput!"
                # call the transition pruning routine to shorten the last input
                debug("Attempting to minimize transition")
                last_input = last_input.flatten(inplace=True)
                # we clone the current path because minimization may corrupt it
                self._last_path = self._current_path.copy()
                try:
                    last_input = await self._minimize_transition(
                        self._last_path, self._current_state, last_input)
                except Exception as ex:
                    # Minimization failed, again probably due to an
                    # indeterministic target
                    warning(f"Minimization failed {ex=}")
                    raise
                last_input = last_input.flatten(inplace=True)
            elif validate:
                try:
                    debug("Attempting to reproduce transition")
                    # we make a copy of _current_path because the loader may
                    # modify it, but at this stage, we may want to restore state
                    # using that path
                    self._last_path = self._current_path.copy()
                    src = await self.reload_state(self._last_state, dryrun=True)
                    assert self._last_state == src
                    await self._driver.execute_input(input)
                    actual_state = self._tracker.peek(
                        src, self._current_state)
                    if self._current_state != actual_state:
                        raise StatePrecisionException(
                        f"The path to {self._current_state} is imprecise")
                except (StateNotReproducibleException,
                        StatePrecisionException):
                    # * StatePrecisionException:
                    #   This occurs when the predecessor state was reached
                    #   through a different path than that used by
                    #   reload_state(). The incremental input thus applied to
                    #   the predecessor state may have built on top of internal
                    #   program state that was not reproduced by reload_state()
                    #   to arrive at the current successor state.
                    #
                    # * StateNotReproducibleException, StatePrecisionException:
                    #   This occurs when the reload_state() encountered an error
                    #   trying to reproduce a state, most likely due to an
                    #   indeterministic target
                    debug(f"Encountered imprecise state ({self._current_state})")
                    raise
                except Exception as ex:
                    warning(f'{ex}')
                    raise

            if minimize or validate:
                debug(f"{self._current_state} is reproducible!")

        self._last_state = self._current_state
        info(f'Transitioned to {self._current_state}')

        return True, unseen, last_input

    async def _minimize_transition(self,
            state_or_path: LoadableTarget[AbstractState, BaseInput],
            dst: AbstractState, input: BaseInput):
        reduced = False
        orig_len = reduced_len = len(input)

        # Phase 1: perform exponential back-off to find effective tail of input
        ValueProfiler('status')('minimize_exp_backoff')
        end = len(input)
        begin = end - 1
        # we do not perform exponential backoff past the midpoint, because that
        # is done in phase 2 anyway
        while begin > end // 2:
            success = True
            src = await self.reload_state(state_or_path, dryrun=True)
            exp_input = input[begin:]
            try:
                await self._driver.execute_input(exp_input)
            except Exception:
                success = False

            success &= dst == self._tracker.peek(src, dst)
            if success:
                reduced = True
                break
            begin = 0 if (diff := end - (end - begin) * 2) < 0 else diff
        reduced_len -= begin

        # Phase 2: prune out dead instructions
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
                ValueProfiler('status')(f'minimize_bin_search'
                    f' (done:{100*i/(2 * (len(input) - begin)):.1f}%'
                    f' reduced:{100*(orig_len-reduced_len)/orig_len:.1f}%'
                    f'={orig_len-reduced_len}/{orig_len})')
                success = True
                src = await self.reload_state(state_or_path, dryrun=True)
                tmp_lin_input = lin_input[:cur] + lin_input[cur + step:]
                try:
                    await self._driver.execute_input(tmp_lin_input)
                except Exception:
                    success = False

                success &= dst == self._tracker.peek(src, dst)
                if success:
                    reduced = True
                    lin_input = tmp_lin_input.flatten(inplace=True)
                    end -= step
                    reduced_len -= step
                else:
                    cur += step
                i += 1
            step //= 2

        # Phase 3: make sure the reduced transition is correct
        src = await self.reload_state(state_or_path, dryrun=True)
        await self._driver.execute_input(lin_input)
        success = (dst == self._tracker.peek(src, dst))
        if not success and reduced:
            # finally, we fall back to validating the original input
            lin_input = input
            src = await self.reload_state(state_or_path, dryrun=True)
            await self._driver.execute_input(lin_input)
            success = (dst == self._tracker.peek(src, dst))
        if not success:
            raise StatePrecisionException("destination state did not match current state")

        if reduced:
            return lin_input
        else:
            return input

class BaseExplorerContext(BaseDecorator):
    """
    This context object provides a wrapper around the input to be sent to the
    target. By wrapping the iterator method, the explorer is updated after
    every instruction in the input. If a state change happens within an input
    sequence, the input is split, where the first part is used as the transition
    and the second part continues to build up state for any following
    transition.
    """
    def __init__(self, explorer: BaseExplorer, **kwargs):
        self._exp = explorer
        self._start = self._stop = None
        self._update_kwargs = kwargs

    def input_gen(self):
        # we delay the call to the slicing decorator until needed
        head = self._exp._last_state.last_input
        tail = self._input[self._start:self._stop]
        if head is None:
            if self._start >= self._stop:
                return None
            else:
                return tail
        else:
            return head + tail

    async def update_state(self, *args, **kwargs):
        await self._exp._state_update_cb(*args, **kwargs)

    async def update_transition(self, *args, **kwargs):
        await self._exp._transition_update_cb(*args, **kwargs)

    @property
    def orig_input(self):
        # we pop the BaseExplorerContext decorator itself to obtain a reference
        # to the original input, typically the output of the input generator
        return self._input.pop_decorator()[0]

    async def ___aiter___(self, input, orig):
        exp = self._exp
        self._start = self._stop = 0
        idx = -1
        for idx, instruction in enumerate(input):
            ValueProfiler('status')('fuzz')
            self._stop = idx + 1
            yield instruction
            # the generator execution is suspended until next() is called so
            # the BaseExplorer update is only called after the instruction
            # is executed by the loader

            try:
                last_state = exp._last_state
                updated, new, last_input = await exp.update(self.input_gen,
                    **self._update_kwargs)
                if updated:
                    self._start = idx + 1

                    # if an update has happened, we've transitioned out of the
                    # last_state and as such, it's no longer necessary to keep
                    # track of the last input
                    last_state.last_input = None
            except Exception as ex:
                self._stop = idx + 1
                # we also clear the last input on an exception, because the next
                # course of action will involve a reload_state
                last_state.last_input = None
                await self.update_state(exp._current_state,
                    breadcrumbs=exp._last_path,
                    input=self.input_gen(), orig_input=self.orig_input, exc=ex)
                raise
            else:
                # FIXME doesn't work as intended, but could save from recalculating cov map diffs if it works
                # self._tracker.update(self._last_state, last_input, peek_result=self._current_state)
                if exp._current_state != exp._tracker.update_state(last_state,
                        input=last_input):
                    raise StabilityException("Failed to obtain consistent behavior",
                        exp._tracker.current_state)

                if updated:
                    exp._tracker.update_transition(
                        last_state, exp._current_state, last_input,
                        state_changed=True)
                    exp._current_path.append(
                        (last_state, exp._current_state, last_input))

                await self.update_state(exp._current_state, input=last_input,
                    orig_input=self.orig_input, breadcrumbs=exp._last_path)
                await self.update_transition(
                    last_state, exp._current_state, last_input,
                    orig_input=self.orig_input, breadcrumbs=exp._last_path,
                    state_changed=updated, new_transition=new)

            # FIXME The explorer interrupts the target to verify individual
            # transitions. To verify a transition, the last state is loaded, and
            # the last input which observed a new state is replayed. However,
            # due to imprecision of states, the chosen path to the last state
            # may result in a different substate, and the replayed input thus no
            # longer reaches the new state. When such an imprecision is met, the
            # target is kept running, but the path has been "corrupted".
            #
            # It may be better to let the input update the explorer freely
            # all throughout the sequence, and perform the validation step at
            # the end to make sure no interesting states are lost.

        if idx >= 0:
            # commit the rest of the input
            exp._last_state.last_input = self.input_gen()

    def ___iter___(self, input, orig):
        return orig()
