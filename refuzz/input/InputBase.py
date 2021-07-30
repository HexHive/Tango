from abc import ABC, abstractmethod

class InputBase(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    # TODO add a transition(self) abstract method, so that each input can
    # generate a transition object relevant to it, and use it in the
    # StateLoaderBase and in the StateManager to populate the state machine