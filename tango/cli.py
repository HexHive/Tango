from . import info, debug

from tango.core import (FuzzerSession, get_profiler, get_all_profilers)
from tango.common import AsyncComponent, get_session_task_group

import asyncio

__all__ = ['CLILogger']

class CLILogger(AsyncComponent, component_type='cli',
        capture_paths=['cli.*'], catch_all=True):
    def __init__(self, *, update_period: float=10, **kwargs):
        super().__init__(**kwargs)
        self._update_period = update_period

    async def run(self):
        instruction_prof = get_profiler('perform_instruction', None)
        if instruction_prof:
            await instruction_prof.listener(
                period=self._update_period)(self.update_stats)

    async def update_stats(self, *args, ret=None, **kwargs):
        msg = 'Stat summary:\n'
        max_width = max(len(name) for name,_ in get_all_profilers())
        fmt = f'{{name:{max_width}}}: {{value}}\n'
        for name, obj in get_all_profilers():
            try:
                msg += fmt.format(name=name, value=obj)
            except Exception:
                pass
        info(msg)