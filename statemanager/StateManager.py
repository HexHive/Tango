from statemanager import StateBase,
                         StateMachine,
                         StateTrackerBase

class StateManager:
    def __init__(self, tracker: StateTrackerBase, scheduler="default"):
        self._tracker = tracker
        self._sm = StateMachine(self._tracker.initial_state)

    @property
    def state_machine(self) -> StateMachine:
        return self._sm

    @property
    def state_tracker(self) -> StateTrackerBase:
        return self._tracker

    def step(self):
        """
        Updates the state queues according to the scheduler.
        May need to receive information about the current state to update it.
        """
        pass