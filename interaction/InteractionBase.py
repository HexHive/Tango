from abc    import ABC
from loader import ChannelBase
from typing import Sequence

class InteractionBase(ABC):
    @abstractmethod
    def perform(self, channel: ChannelBase) -> Sequence[InteractionBase]:
        pass

    @abstractmethod
    def mutate(self, *args, **kwargs) -> InteractionBase:
        pass