from statemanager import TransitionBase
from statemanager import GrammarState
from input import InputBase

def GrammarTransition(TransitionBase):

    """
    A transition will have the following members as part of the transition mechanism
    Initial state
    Final state
    Input - Input is actually the name of the DataModel that this transition refers to.


    Input - It is very complex to define the input class in such a way as to elicit the structure...
    We can define the input tokens here, and use them in the later part.
    Also, any attribute defined with mutable False is a token/keyword for the protocol.
    """

    def __init__(self, inital_state: GrammarState, final_state: GrammarState, input: str):
        # TODO
        self.initial_state = initial_state
        self.final_state = final_state
        self._input = input


    def __add__(self, other):
        # TODO: Not sure what add exactly is
        pass

    def input(self) -> InputBase:
        # TODO: Generate a new input from the tree. Use _input to identify the correct node
        pass