from abc import abstractmethod
import profiler
from profiler import ProfilerBase, ProfilingTasks
from asyncio import wait_for, shield, create_task, TimeoutError, CancelledError

class PeriodicProfiler(ProfilerBase):
    def __init__(self, name, period=1, **kwargs):
        super().__init__(name, **kwargs)
        if not self._init_called:
            self._period = period

    def ___call___(self, obj):
        if not self._init_called:
            ProfilingTasks.append(self._task_worker())
        return obj

    async def _task_worker(self):
        while True:
            try:
                await wait_for(shield(profiler.ProfilingStoppedEvent.wait()), self._period)
            except TimeoutError:
                self._task()
            except CancelledError:
                return

    @abstractmethod
    def _task(self):
        pass