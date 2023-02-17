from abc import ABC, abstractmethod
from typing import Union
from input import AbstractInput
from statemanager import StateMachine
from tracker import AbstractState

class AbstractStrategy(ABC):
    @abstractmethod
    def update_state(self, state: AbstractState, *, input: AbstractInput, exc: Exception=None, **kwargs):
        """
        Updates the internal strategy parameters related to the state. In case a
        state is invalidated, it should remain so until it is revalidated in a
        following call to update_state(). Otherwise, invalidated states must not
        be selected at the target state.

        :param      state:  The current state of the target.
        :type       state:  AbstractState
        :param      exc:    An exception that occured while processing the input.
        :type       exc:    Exception
        """
        pass

    @abstractmethod
    def update_transition(self, source: AbstractState, destination: AbstractState, input: AbstractInput, *, state_changed: bool, exc: Exception=None, **kwargs):
        """
        Similar to update_state(), but for transitions.

        :param      source:       The source state of the transition.
        :type       source:       AbstractState
        :param      destination:  The destination state. This can be assumed to
                                  be the current state of the target too.
        :type       destination:  AbstractState
        :param      input:        The input associated with the transition.
        :type       input:        AbstractInput
        :param      exc:          An exception that occured while processing the
                                  input.
        :type       exc:          Exception
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
    def target_state(self) -> AbstractState:
        """
        The selected target state. This is mainly used for reporting purposes.
        """
        pass

    @property
    @abstractmethod
    def target(self) -> Union[AbstractState, list]:
        """
        The last selected target state or path.
        """
        pass