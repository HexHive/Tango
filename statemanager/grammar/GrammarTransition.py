from statemanager import TransitionBase
from statemanager import GrammarState
from input import InputBase

def GrammarTransition(TransitionBase):

    """
    A transition will have the following members as part of the transition mechanism
    Initial state
    Final state
    Input
    """

    def __init__(self, inital_state: GrammarState, final_state: GrammarState, input: InputBase):
        # TODO
        self.initial_state = initial_state
        self.final_state = final_state
        self.input = input


    def __add__(self, other):
        # TODO: Not sure what add exactly is
        pass

    def input(self) -> InputBase:
        # TODO: To be designed in a way to make grammar given input into genering input
        pass