from abc import ABC, abstractmethod

class InputBase(ABC):
    @abstractmethod
    def __iter__(self):
        pass