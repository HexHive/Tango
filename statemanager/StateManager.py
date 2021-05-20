from statemanager import (StateBase,
                         StateMachine,
                         StateTrackerBase,
                         TransitionBase)
from loader       import StateLoaderBase

class StateManager:
    def __init__(self, loader: StateLoaderBase, tracker: StateTrackerBase,
            scheduler="default"):
        self._loader = loader
        self._tracker = tracker
        self._last_state = self._tracker.initial_state
        self._sm = StateMachine(self._last_state)

    @property
    def state_machine(self) -> StateMachine:
        return self._sm

    @property
    def state_tracker(self) -> StateTrackerBase:
        return self._tracker

    def reset_state(self):
        self._loader.load_state(self._tracker.initial_state, self)

    def step(self) -> bool:
        """
        Updates the state queues according to the scheduler.
        May need to receive information about the current state to update it.

        :returns:   Whether or not the step resulted in a state change
        :rtype:     bool
        """
        pass

    def update(self, transition: TransitionBase) -> bool:
        """
        Updates the state machine in case of a state change.

        :param      transition:  The transition that may have resulted in the
                      state change
        :type       transition:  TransitionBase

        :returns:   Whether or not a new state was reached.
        :rtype:     bool
        """

        updated = False
        current_state = self._tracker.current_state
        if current_state != self._last_state:
            self._sm.add_transition(self._last_state, current_state, transition)
            self._last_state = current_state
            updated = True

        # update the current state (e.g., if it needs to track interesting cov)
        current_state.update(self, transition)
        return updated