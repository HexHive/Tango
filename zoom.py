from tangofuzz import *
from fuzzer import FuzzerSession, FuzzerConfig
import logging
from common import Suspendable
import asyncio

logging.getLogger("interaction").setLevel(logging.DEBUG)
logging.getLogger("common").setLevel(logging.DEBUG)
logging.getLogger("generator").setLevel(logging.DEBUG)

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

async def move_in_square():
    state = sess._sman._tracker.current_state
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
            await interaction.ReachInteraction(state, target, stop_at_target=True).perform(chan)

async def suspend_the_mofuka():
    while True:
        await asyncio.sleep(0.1)
        logging.warning("Suspending now!")
        asyncio.get_running_loop().main_task.coro.suspend()
        await asyncio.sleep(0.1)
        logging.warning("Resuming now!")
        asyncio.get_running_loop().main_task.coro.resume()

keys = []
def suspend_cb():
    async def async_suspend_cb():
        global keys
        res = await sess._loader._channel.clear()
        keys.extend(res)
    asyncio.get_running_loop().create_task(async_suspend_cb())

def resume_cb():
    async def async_resume_cb():
        global keys
        await sess._loader._channel.send(keys)
    asyncio.get_running_loop().create_task(async_resume_cb())

async def main():
    pauser_task = asyncio.create_task(suspend_the_mofuka())
    await move_in_square()

async def bootstrap():
    await sess.initialize()
    main_task = asyncio.get_running_loop().main_task = asyncio.current_task()
    main_task.suspendable_ancestors = []
    main_task.coro = Suspendable(main(), suspend_cb=suspend_cb, resume_cb=resume_cb)
    await main_task.coro

asyncio.run(bootstrap())

