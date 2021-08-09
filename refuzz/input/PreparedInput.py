from input       import InputBase
from interaction import InteractionBase
from typing      import Iterable

class PreparedInput(InputBase):
    """
    A buffered input. All interactions are readily available and can be exported
    to a file.
    """
    def __init__(self, interactions: Iterable[InteractionBase]=None):
        self._interactions = []
        if interactions:
            self._interactions.extend(interactions)

    def append(self, interaction: InteractionBase):
        self._interactions.append(interaction)

    def extend(self, interactions: Iterable[InteractionBase]):
        self._interactions.extend(interactions)

    def export(self, type="pcap"):
        raise NotImplemented()

    def ___iter___(self):
        return iter(self._interactions)
