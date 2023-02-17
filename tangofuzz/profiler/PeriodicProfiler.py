from abc import abstractmethod
import profiler
from profiler import AbstractProfiler, ProfilingTasks
from asyncio import wait_for, TimeoutError, CancelledError

class PeriodicProfiler(AbstractProfiler):
    def __init__(self, *, period=1, **kwargs):
        super().__init__(**kwargs)
        self._period = period
        ProfilingTasks.append(self._task_worker())

    async def _task_worker(self):
        while True:
            try:
                await wait_for(profiler.ProfilingStoppedEvent.wait(), self._period)
                return
            except TimeoutError:
                self.do_task()
            except CancelledError:
                return

    @abstractmethod
    def do_task(self):
        pass
