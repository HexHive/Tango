from input        import PreparedInput
from statemanager import TransitionBase

class PreparedTransition(TransitionBase):
    def __init__(self, input: PreparedInput):
        self._input = input

    def __add__(self, other: PreparedTransition):
        return PreparedTransition(
            PreparedInput(
                self._input._interactions + other._input._interactions
            )
        )

    @property
    def input(self) -> PreparedInput:
        return self._input