from input       import BaseInput
from interaction import AbstractInteraction
from typing      import Sequence

class PreparedInput(BaseInput):
    """
    A buffered input. All interactions are readily available and can be exported
    to a file.
    """
    def __init__(self, *, interactions: Sequence[AbstractInteraction]=None, **kwargs):
        super().__init__(**kwargs)
        self._interactions = []
        if interactions:
            self._interactions.extend(interactions)

    def append(self, interaction: AbstractInteraction):
        self._interactions.append(interaction)

    def extend(self, interactions: Sequence[AbstractInteraction]):
        self._interactions.extend(interactions)

    def ___iter___(self):
        return iter(self._interactions)

    def ___len___(self):
        return len(self._interactions)

    # def __add__(self, other: PreparedInput):
    #     return PreparedInput(
    #         self._interactions + other._interactions
    #     )

    # def __eq__(self, other: PreparedInput):
    #     return self._interactions == other._interactions

    # def __getitem__(self, key):
    #     if isinstance(key, slice):
    #         return PreparedInput(self._interactions.__getitem__(key))
    #     else:
    #         return PreparedInput([self._interactions[key],])
