from abc import ABC, abstractmethod

class StrategyBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self):
        pass