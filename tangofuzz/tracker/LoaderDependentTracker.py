from tracker import StateTrackerBase
from loader import StateLoaderBase

class LoaderDependentTracker(StateTrackerBase):
    def __init__(self, /, *, loader: StateLoaderBase, **kwargs):
        super().__init__(**kwargs)
        self._loader = loader