from __future__ import annotations
from . import debug, info, warning, critical

from tango.exceptions import (StabilityException, StateNotReproducibleException,
    StatePrecisionException)
from tango.core.tracker import AbstractState, AbstractTracker
from tango.core.input import (AbstractInput, BaseInput, PreparedInput,
    BaseDecorator, EmptyInput)
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

        # these are generally persistent until a reload_state
        self._current_path = []
        self._accumulated_input = EmptyInput()
        self._last_state = None

    async def finalize(self, owner: ComponentOwner):
        self._last_state = self._tracker.entry_state
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
        except StopAsyncIteration:
            return current

    @FrequencyProfiler('resets')
    async def reload_state(self, state_or_path: LoadableTarget=None, *,
            dryrun=False) -> AbstractState:
        if not dryrun:
            ValueProfiler('status')('reset_state')
            self._reset_current_path()
            self._accumulated_input = EmptyInput()

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
        return BaseExplorerContext(input, explorer=self, **kwargs)

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
            -> tuple(bool, bool, AbstractState, BaseInput):
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
            return False, False, None, None

        debug(f"Reached {current_state = }")
        current_input = input_gen()

        # FIXME this logic should be moved to the tracker; it is more suitable
        # for judging whether a state or a transition is new
        if current_state == self._last_state:
            return False, False, current_state, current_input

        unseen = current_state not in self._tracker.state_graph.successors(self._last_state)
        debug(f"Possible transition from {self._last_state} to {current_state}")

        if unseen:
            if minimize:
                # During minimization, the input is sliced/joined/cached, which
                # are facilities provided by BaseInput.
                # An input type that does not support these capabilities must
                # inherit from AbstractInput and disable minimization (or use a
                # different Explorer).
                assert isinstance(current_input, BaseInput), \
                    "Minimizing requires inputs to be BaseInput!"
                # call the transition pruning routine to shorten the last input
                debug("Attempting to minimize transition")
                current_input = current_input.flatten()
                # we clone the current path because minimization may corrupt it
                last_path = self._current_path.copy()
                try:
                    current_input = await self._minimize_transition(
                        last_path, current_state, current_input)
                    # we update the reference to the current state, in case
                    # minimization invalidated it
                    current_state = self._tracker.current_state
                except Exception as ex:
                    # Minimization failed, again probably due to an
                    # indeterministic target
                    warning(f"Minimization failed {ex=}")
                    raise
                current_input = current_input.flatten()
            elif validate:
                try:
                    debug("Attempting to reproduce transition")
                    # we make a copy of _current_path because the loader may
                    # modify it, but at this stage, we may want to restore state
                    # using that path
                    last_path = self._current_path.copy()
                    src = await self.reload_state(last_path, dryrun=True)
                    assert self._last_state == src
                    await self._loader.apply_transition(
                        (src, current_state, current_input), src,
                        update_cache=False)
                except StateNotReproducibleException:
                    # * StateNotReproducibleException:
                    #   This occurs when the reload_state() encountered an error
                    #   trying to reproduce a state, most likely due to an
                    #   indeterministic target
                    warning(f"Encountered indetermenistic state ({current_state})")
                    raise
                except (StabilityException, StatePrecisionException) as ex:
                    # * StatePrecisionException:
                    #   This occurs when the predecessor state was reached
                    #   through a different path than that used by
                    #   reload_state(). The incremental input thus applied to
                    #   the predecessor state may have built on top of internal
                    #   program state that was not reproduced by reload_state()
                    #   to arrive at the current successor state.
                    warning(f"Validation failed {ex=}")
                    raise StatePrecisionException(
                            f"The path to {current_state} is imprecise") \
                        from ex
                except Exception as ex:
                    warning(f'{ex}')
                    raise

            if minimize or validate:
                debug(f"{current_state} is reproducible!")

        self._last_state = current_state
        return True, unseen, current_state, current_input

    async def _minimize_transition(self, state_or_path: LoadableTarget,
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
                await self._loader.apply_transition((src, dst, exp_input), src,
                    update_cache=False)
            except Exception as ex:
                debug(f'{ex=}')
                success = False

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
                    await self._loader.apply_transition(
                        (src, dst, tmp_lin_input), src,
                        update_cache=False)
                except Exception as ex:
                    debug(f'{ex=}')
                    success = False

                if success:
                    reduced = True
                    lin_input = tmp_lin_input.flatten()
                    end -= step
                    reduced_len -= step
                else:
                    cur += step
                i += 1
            step //= 2

        # Phase 3: make sure the reduced transition is correct
        src = await self.reload_state(state_or_path, dryrun=True)
        success = True
        try:
            await self._loader.apply_transition((src, dst, lin_input), src,
                update_cache=False)
        except Exception as ex:
            debug(f'{ex=}')
            success = False
            if reduced:
                # finally, we fall back to validating the original input
                src = await self.reload_state(state_or_path, dryrun=True)
                await self._loader.apply_transition((src, dst, input), src,
                    update_cache=False)
                reduced = False
                success = True

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
    def __init__(self, input: AbstractInput, /, *, explorer: BaseExplorer, **kw):
        super().__init__(input)
        self._exp = explorer
        self._start = self._stop = None
        self._update_kwargs = kw

    def input_gen(self):
        # we delay the call to the slicing decorator until needed
        head = self._exp._accumulated_input
        tail = self[self._start:self._stop]
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
        return self.pop()

    async def __aiter__(self, *, orig):
        exp = self._exp
        self._start = self._stop = 0
        idx = -1
        for idx, instruction in enumerate(self._orig):
            ValueProfiler('status')('fuzz')
            self._stop = idx + 1
            yield instruction
            # the generator execution is suspended until next() is called so
            # the BaseExplorer update is only called after the instruction
            # is performed by the driver

            try:
                last_state = exp._last_state
                updated, unseen, tmp_state, current_input = await exp.update(
                    self.input_gen, **self._update_kwargs)
                if updated:
                    self._start = idx + 1

                    # if an update has happened, we've transitioned out of the
                    # last_state and as such, it's no longer necessary to keep
                    # track of the last input
                    exp._accumulated_input = EmptyInput()
            except Exception as ex:
                self._stop = idx + 1
                await self.update_state(exp._tracker.current_state,
                    breadcrumbs=exp._current_path,
                    input=self.input_gen(), orig_input=self.orig_input,
                    exc=ex)
                raise
            else:
                # FIXME doesn't work as intended, but could save from
                # recalculating cov map diffs if it works
                # exp._tracker.update(last_state, current_input,
                # peek_result=current_state)
                current_state = exp._tracker.update_state(last_state,
                    input=current_input)
                if tmp_state != current_state:
                    raise StabilityException(
                        "Failed to obtain consistent behavior",
                        current_state)

                breadcrumbs = exp._current_path.copy()
                if updated:
                    info(f'Transitioned to {current_state}')
                    exp._tracker.update_transition(
                        last_state, current_state, current_input,
                        state_changed=True)
                    exp._current_path.append(
                        (last_state, current_state, current_input))

                await self.update_state(current_state, input=current_input,
                    orig_input=self.orig_input, breadcrumbs=breadcrumbs)
                await self.update_transition(
                    last_state, current_state, current_input,
                    orig_input=self.orig_input, breadcrumbs=breadcrumbs,
                    state_changed=updated, new_transition=unseen)

        if idx >= 0:
            # commit the rest of the input
            exp._accumulated_input = self.input_gen()

    def __iter__(self, *, orig):
        return orig()