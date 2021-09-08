from __future__ import annotations
from . import debug

from typing import Callable
from statemanager import (StateBase,
                         StateMachine,
                         StateTrackerBase)
from input        import InputBase, CachingDecorator
from loader       import StateLoaderBase

class StateManager:
    class StateManagerContext:
        def __init__(self, sman: StateManager, input: InputBase):
            self._sman = sman
            self._input = input
            self._start = self._stop = None

        def input_gen(self):
            # we use a generator so as not to call the slicing decorator
            # needlessly
            return self._input[self._start:self._stop]

        def __iter__(self):
            self._start = self._stop = 0
            for idx, interaction in enumerate(self._input):
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
                # we deepcopy the generated input so as not to have any
                # side-effects
                self._sman._last_state.last_input = self.input_gen()[:]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            pass

    def __init__(self, startup_input: InputBase, loader: StateLoaderBase,
            tracker: StateTrackerBase, scheduler: str, cache_inputs: bool):
        self._loader = loader
        self._tracker = tracker
        self._cache_inputs = cache_inputs

        self._last_state = self._tracker.entry_state
        self._startup_input = startup_input
        self._sm = StateMachine(self._last_state)

        self._counter = 0

    @property
    def state_machine(self) -> StateMachine:
        return self._sm

    @property
    def state_tracker(self) -> StateTrackerBase:
        return self._tracker

    def reset_state(self, state=None):
        if self._last_state:
            # must clear last state's input whenever a new state is loaded
            # WARN if using SnapshotStateLoader, the snapshot must be taken when
            # the state is first reached, not when a transition is about to
            # happen. Otherwise, last_input may be None, but the snapshot may
            # have residual effect from previous inputs before the state change.
            self._last_state.last_input = None
            # we also set _last_state to the entry state in case the loader
            # needs to execute inputs to reach the target state (e.g. ReplayStateLoader)
            # so that no new edges are added between the last state and the entry state
            self._last_state = self._tracker.entry_state

        if state is None:
            self._loader.load_state(self._tracker.entry_state, self)
            self._last_state = self._tracker.entry_state

            self._loader.execute_input(self._startup_input, self)
        else:
            self._loader.load_state(state, self)
            self._last_state = state

    def get_context(self, input: InputBase) -> StateManager.StateManagerContext:
        return self.StateManagerContext(self, input)

    def step(self) -> bool:
        """
        Updates the state queues according to the scheduler.
        May need to receive information about the current state to update it.

        :returns:   Whether or not the step resulted in a state change
        :rtype:     bool
        """
        self._counter += 1
        self._counter %= 5000

        if self._counter == 0:
            import random
            new_state = random.sample(self._sm._graph.nodes, k=1)[0]
            self.reset_state(new_state)
            debug(f'Reset state to {new_state=}')

        # self.reset_state()
        pass

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
        self._tracker.update_state(self._last_state, current_state, input_gen)

        # the tracker may return None as current_state, in case it has not yet
        # finished the training phase (preprocessing seeds)
        if current_state is not None and current_state != self._last_state:
            # TODO optionally call the transition pruning routine to shorten the
            # last input
            if self._last_state.last_input is not None:
                last_input = self._last_state.last_input + input_gen()
            else:
                last_input = input_gen()

            if self._cache_inputs:
                last_input = CachingDecorator()(last_input, copy=False)

            self._sm.update_transition(self._last_state, current_state,
                last_input)
            self._last_state = current_state
            debug(f'Transitioned to {current_state=}')
            updated = True

        return updated