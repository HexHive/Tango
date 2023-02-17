from tracker import AbstractStateTracker
from loader import AbstractStateLoader

class LoaderDependentTracker(AbstractStateTracker):
    def __init__(self, /, *, loader: AbstractStateLoader, **kwargs):
        super().__init__(**kwargs)
        self._loader = loader