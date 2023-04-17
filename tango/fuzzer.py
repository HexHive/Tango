from . import info, warning

from tango.core import (FuzzerConfig, FuzzerSession,
    initialize as initialize_profiler, TimeElapsedProfiler, is_profiling_active)
from tango.common import (Suspendable,
    create_session_context, get_session_task_group)
from tango.webui import WebRenderer

from aioconsole import AsynchronousConsole, get_standard_streams
from ast import literal_eval
import asyncio
import argparse
import logging
import signal
import sys

class Fuzzer:
    def __init__(self, args=None):
        self._argspace = self.parse_args(args)
        self.configure_verbosity(self._argspace.verbose)
        self._overrides = self.construct_overrides(self._argspace.override)
        self._sessions = []
        self._cleanup = False

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser(description=(
            "Launches a TangoFuzz fuzzing session."
        ))
        parser.add_argument("config",
            help="The path to the TangoFuzz fuzz.json file.")
        parser.add_argument('--override', '-o', action='append', nargs=2)
        parser.add_argument('--sessions', '-s', type=int, default=1,
            help="The number of concurrent fuzzing sessions to run.")
        parser.add_argument('-v', '--verbose', action='count', default=0,
            help=("Controls the verbosity of messages. "
                "-v prints info. -vv prints debug. Default: warnings and higher.")
            )
        return parser.parse_args(args)

    @staticmethod
    def configure_verbosity(level, *, logger=None):
        mapping = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
        # will raise exception when level is invalid
        numeric_level = mapping[level]
        logging.getLogger(logger).setLevel(numeric_level)

    @staticmethod
    def construct_overrides(override_list: list[tuple[str, str]]) -> dict:
        def is_number_repl_isdigit(s):
            """ Returns True if string is a number. """
            """https://stackoverflow.com/a/23639915 """
            return s.lstrip('-') \
                .replace('.','',1) \
                .replace('e-','',1) \
                .replace('e','',1) \
                .isdigit()
        if not override_list:
            return
        overrides = dict()
        for name, value in override_list:
            keys = name.split('.')
            levels = keys[:-1]
            d = overrides
            for k in levels:
                if not d.get(k):
                    d[k] = dict()
                d = d[k]
            key = keys[-1]
            if value.lower() in ('true', 'false'):
                value = (value == 'true')
            elif is_number_repl_isdigit(value):
                value = literal_eval(value)
            d[key] = value
        return overrides

    def _loop_exception_handler(self, loop, context):
        import ipdb; ipdb.set_trace()
        pass

    async def _bootstrap(self):
        # use session-local task contexts
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(self._loop_exception_handler)

        self._stream_cache = {}
        # we use an empty tuple for `streams` to prevent it being seen as None,
        # in which case, aioconsole would generate its own streams
        self._repl = AsynchronousConsole(streams=(), loop=loop,
            locals=locals() | {
                'exit': lambda code=0: self._cleanup_and_exit(loop, code),
                'loop': loop
            })

        # set up the main suspendable fuzzing task
        self._suspendable = Suspendable(self.run_all_sessions())

        # time starts now
        self._timer = TimeElapsedProfiler('elapsed', session_local=False)

        try:
            # let the fuzzing begin
            self._runner = asyncio.create_task(self._suspendable.as_coroutine())
            self._bootstrap_sigint(loop, handle=True)
            await self._runner
        except asyncio.CancelledError:
            pass
        except ExceptionGroup as exg:
            warning("Exceptions encountered while terminating sessions:\n"
                f"{chr(10).join(repr(x) for x in exg.exceptions)}"
            )
        except Exception as ex:
            warning(f"Exception raised while attempting to terminate: {ex}")
        info("Goodbye!")

    async def run_all_sessions(self):
        try:
            async with asyncio.TaskGroup() as tg:
                await initialize_profiler(tg)
                for i in range(self._argspace.sessions):
                    context = create_session_context(tg)
                    tg.create_task(
                        self.create_session(i),
                        name=f'Session-{i}',
                        context=context)
        except ExceptionGroup as exg:
            import ipdb; ipdb.set_trace()
            pass

    async def create_session(self, sid):
        name = asyncio.current_task().get_name()
        config = FuzzerConfig(self._argspace.config, self._overrides)
        session = await config.instantiate('session', sid)
        self._sessions.append(session)
        tg = get_session_task_group()
        tg.create_task(Suspendable(session.run()).as_coroutine(), name=name)
        if is_profiling_active():
            webui = await config.instantiate('webui')
            tg.create_task(webui.run(), name=f'webui[{name}]')

    def _bootstrap_sigint(self, loop, handle=True):
        if handle:
            loop.add_signal_handler(signal.SIGINT, self._sigint_handler, loop)
        else:
            loop.remove_signal_handler(signal.SIGINT)

    def _sigint_handler(self, loop):
        async def await_repl(task, restore):
            await task
            self._suspendable.set_tasks(wakeup_task=restore)
        repl_task = loop.create_task(self._interact(loop))
        _, wakeup_restore = self._suspendable.tasks
        self._suspendable.set_tasks(
            wakeup_task=await_repl(repl_task, wakeup_restore))

    async def _interact(self, loop):
        # disable handler
        self._bootstrap_sigint(loop, handle=False)

        # pause fuzzing timer
        self._timer()

        # suspend fuzzing sessions (and call respective suspend handlers)
        self._suspendable.suspend()

        # The coroutine get_standard_streams is deliberately not awaited.
        # AsynchronousConsole supports having an awaitable `streams` attribute
        # that is called every time `interact` is invoked. To allow for reusing
        # the console, we use this coroutine with a no-op cache so that streams
        # are re-generated every time `interact` is called.
        # interact.
        # Unfortunately, this also means the coroutine is consumed after being
        # awaited, and we need to recreate it again before every call.
        self._stream_cache.clear()
        self._repl.streams = get_standard_streams(cache=self._stream_cache, use_stderr=True, loop=loop)
        await self._repl.interact(banner="Fuzzing paused (type exit() to quit)",
            stop=False, handle_sigint=False)

        self._suspendable.resume()

        if not self._cleanup:
            # reopen stdin if it is at EOF
            if self._repl.reader.at_eof():
                sys.stdin.close()
                sys.stdin = open("/dev/tty")

            self._timer()
            self._bootstrap_sigint(loop, handle=True)

    def _cleanup_and_exit(self, loop, code, /):
        self._cleanup = True
        self._runner.cancel()
        sys.exit(code)

    def run(self):
        # child watcher prints a warning when its children are reaped by
        # someone else. See FIXME in WebDataLoader
        logging.getLogger('asyncio').setLevel(logging.CRITICAL)
        asyncio.run(self._bootstrap())