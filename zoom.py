from tangofuzz import *
from fuzzer import FuzzerSession, FuzzerConfig
import logging

logging.getLogger("interaction").setLevel(logging.DEBUG)

# ch = networkio.X11ChannelFactory(timescale=1).create(program_name="chocolate-doom")

# while True:
#     interaction.MoveInteraction("forward").perform(ch)
#     interaction.MoveInteraction("rotate_left").perform(ch)
#     interaction.DelayInteraction(0.5).perform(ch)
#     interaction.ShootInteraction().perform(ch)
#     interaction.DelayInteraction(0.5).perform(ch)
#     interaction.MoveInteraction("rotate_left", stop=True).perform(ch)
#     interaction.MoveInteraction("forward", stop=True).perform(ch)

config = FuzzerConfig("./targets/zoom/fuzz.json")
sess = FuzzerSession(config)

state = sess._sman._tracker.current_state
chan = sess._loader
chan = sess._loader._channel
struct = state.state_manager._tracker._reader.struct

initial_pos = (struct.x, struct.y)
offsets = (
    (80, 50),
    (80, 550),
    (-150, 550),
    (-150, 50),
)

while True:
    for o in offsets:
        target = (initial_pos[0] + o[0], initial_pos[1] + o[1])
        interaction.ReachInteraction(state, target, stop_at_target=True).perform(chan)
