from abc import ABC, abstractmethod
from typing import Union
from input import InputBase
from statemanager import StateBase, StateMachine

class StrategyBase(ABC):
    def __init__(self, sm: StateMachine, entry_state: StateBase):
        self._sm = sm
        self._entry = entry_state

    @abstractmethod
    def update_state(self, state: StateBase, invalidate: bool, is_new: bool):
        """
        Updates the internal strategy parameters related to the state. In case a
        state is invalidated, it should remain so until it is revalidated in a
        following call to update_state(). Otherwise, invalidated states must not
        be selected at the target state.

        :param      state:       The current state of the target.
        :type       state:       StateBase
        :param      invalidate:  Whether or not the state should be invalidated.
        :type       invalidate:  bool
        :param      is_new:      Whether or not the state was newly discovered
        :type       is_new:      bool
        """
        pass

    @abstractmethod
    def update_transition(self, source: StateBase, destination: StateBase, input: InputBase, invalidate: bool):
        """
        Similar to update_state(), but for transitions.

        :param      source:       The source state of the transition.
        :type       source:       StateBase
        :param      destination:  The destination state. This can be assumed to
                                  be the current state of the target too.
        :type       destination:  StateBase
        :param      input:        The input associated with the transition.
        :type       input:        InputBase
        :param      invalidate:   Whether or not the transition should be
                                  invalidated.
        :type       invalidate:   bool
        """
        pass

    @abstractmethod
    def step(self) -> bool:
        """
        Selects the target state according to the strategy parameters and
        returns whether or not a reset is needed.

        :returns:   Whether or not a state reset is needed.
        :rtype:     bool
        """
        pass

    @property
    @abstractmethod
    def target_state(self) -> StateBase:
        """
        The selected target state. This is mainly used for reporting purposes.
        """
        pass

    @property
    @abstractmethod
    def target(self) -> Union[StateBase, list]:
        """
        The last selected target state or path.
        """
        pass