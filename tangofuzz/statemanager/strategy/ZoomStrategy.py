from statemanager import StateBase, RandomStrategy, StateMachine
from random import Random
from profiler import ProfileValue
from input import InputBase, ZoomInput
from interaction import ReachInteraction

class ZoomStrategy(RandomStrategy):
    def update_transition(self, source: StateBase, destination: StateBase, input: InputBase, invalidate: bool = False):
        src_x, src_y, src_z = map(lambda c: getattr(source._struct, c), ('x', 'y', 'z'))
        dst_x, dst_y, dst_z = map(lambda c: getattr(destination._struct, c), ('x', 'y', 'z'))
        if dst_z > src_z or abs(src_z - dst_z) <= 16:
            # FIXME this interaction is not used, since it is replaced by the loader
            inp = ZoomInput([ReachInteraction(destination, (src_x, src_y, src_z))])
            self._sm.update_transition(destination, source, inp)