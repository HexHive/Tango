Only seems to work with -DSKIP_COUNTS in coverage.c and COVERAGE=func.

This is most likely due to the async event loop they implement: the main program
schedules tasks and polls them for completion in a loop. While waiting for a
client to connect, the event loop could schedule other tasks, and thus execute
other basic blocks and edges. When replaying the same sequence of inputs from a
client, those basic block and edge counts could differ between runs, because the
client is not always guaranteed to connect and send its data at the same time.

This is subject to change in future iterations when states are not directly
related to coverage maps.