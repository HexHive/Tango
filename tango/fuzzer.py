from . import info, warning, error
from tango.exceptions import NotSyncedException, ProcessCrashedException, \
    ProcessTerminatedException, StateNotReproducibleException
from tango.core import (FuzzerConfig, FuzzerSession,
    initialize as initialize_profiler, TimeElapsedProfiler, is_profiling_active,
    get_current_session)
from tango.common import (Suspendable,
    create_session_context, get_session_task_group)
from tango.webui import WebRenderer

from aioconsole import AsynchronousCli, get_standard_streams
from ast import literal_eval
from pathlib import Path
import asyncio
import argparse
import logging
import signal
import sys
import os

class Fuzzer:
    def __init__(self, args=None, overrides={}):
        self._argspace = self.parse_args(args)
        self.configure_verbosity(self._argspace.verbose)
        overrides = list(overrides.items()) + self._argspace.override
        self._overrides = self.construct_overrides(overrides)
        self._sessions = {}
        self._cleanup = False
        if self._argspace.pid:
            Path(self._argspace.pid).write_text(str(os.getpid()))
        if self._argspace.show_pcs:
            os.environ["SHOW_PCS"] = "1"
        if self._argspace.show_syscalls:
            os.environ["SHOW_SYSCALLS"] = "1"

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser(description=(
            "Launches a TangoFuzz fuzzing session."
        ))
        parser.add_argument("config",
            help="The path to the TangoFuzz fuzz.json file.")
        parser.add_argument('--override', '-o', action='append', default=[],
            nargs=2)
        parser.add_argument('--pid', '-p',
            help="Save the fuzzer's process PID at the specified path.")
        parser.add_argument('--sessions', '-s', type=int, default=1,
            help="The number of concurrent fuzzing sessions to run.")
        parser.add_argument('-v', '--verbose', action='count', default=-1,
            help=("Control the verbosity of messages. "
                "-v prints info. -vv prints debug. Default: warnings and higher."))
        parser.add_argument('--show_pcs', action='store_true', default=False,
            help="Show pcs to debug StabilityException.")
        parser.add_argument('--show_syscalls', action='store_true', default=False,
            help="Show syscalls to understand the target.")
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
            if isinstance(value, str):
                if value.lower() in ('true', 'false'):
                    value = (value == 'true')
                elif is_number_repl_isdigit(value):
                    value = literal_eval(value)
            d[key] = value
        return overrides

    def _loop_exception_handler(self, loop, context):
        error(context)
        import traceback
        print(traceback.format_exc())

    async def _bootstrap(self):
        # use session-local task contexts
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(self._loop_exception_handler)

        # set up the main suspendable fuzzing task
        self._suspendable = Suspendable(self.run_all_sessions())

        # time starts now
        self._timer = TimeElapsedProfiler('time_elapsed', session_local=False)

        try:
            # let the fuzzing begin
            self._runner = asyncio.create_task(self._suspendable.as_coroutine())
            self._bootstrap_sigint(loop, handle=True)
            loop.add_signal_handler(
                signal.SIGTERM, self._cleanup_and_exit, loop)
            info("Launching all sessions")
            await self._runner
        # all unhandled expections after the instantiating go here
        except asyncio.CancelledError:
            pass
        except ExceptionGroup as exg:
            warning("Exceptions encountered while terminating sessions:\n%s",
                    chr(10).join(repr(x) for x in exg.exceptions))
        except Exception as ex:
            warning("Exception raised while attempting to terminate: %s", ex)
            import traceback; print(traceback.format_exc())
        info("Goodbye!")

    async def run_all_sessions(self):
        try:
            async with asyncio.TaskGroup() as tg:
                info(f"Initializing profilers into {tg}")
                await initialize_profiler(tg)
                for i in range(self._argspace.sessions):
                    info(f"Launching session {i} with {tg}")
                    await self.launch_session(tg, sid=i)
                    info(f"Launched session {i} with {tg}")
        except ExceptionGroup as exg:
            # all exceptions during instantiating go here
            # trick: disable ASAN for more debugging information
            for ex in exg.exceptions:
                from json import JSONDecodeError
                if isinstance(ex, FileNotFoundError):
                    error(f"Cannot find {ex.filename}")
                elif isinstance(ex, JSONDecodeError):
                    error(f"Cannot decode {ex.doc}, {ex}")
                elif isinstance(ex, ProcessTerminatedException):
                    if isinstance(ex, ProcessCrashedException):
                        error(f"Run into an early crash")
                elif isinstance(ex, StateNotReproducibleException):
                    import ipdb; ipdb.set_trace()
                    error(f"{ex}")
                elif isinstance(ex, NotSyncedException):
                    error(f"{ex}")
                else:
                    raise ex

    async def launch_session(self, tg, *args, **kwargs):
        context = create_session_context(tg)
        info(f"Creating and waiting task create_session with {tg}")
        session = await tg.create_task(
            self.create_session(context, *args, **kwargs),
            context=context)
        info(f"Creating and waiting task start_current_session with {tg}")
        return await tg.create_task(
            self.start_current_session(),
            name=f'Session-{session.id}',
            context=context)

    async def start_current_session(self):
        session = get_current_session()
        tg = get_session_task_group()
        name = asyncio.current_task().get_name()
        tg.create_task(Suspendable(session.run()).as_coroutine(), name=name)
        info(f"Creating task {name} with {tg}")
        if is_profiling_active('webui'):
            info(f"Instantiating webui with {session.owner}")
            webui = await session.owner.instantiate('webui')
            info(f"Creating task webui[{name}] with {tg}")
            tg.create_task(webui.run(), name=f'webui[{name}]')
        if is_profiling_active('cli'):
            info(f"Instantiating cli with {session.owner}")
            cli = await session.owner.instantiate('cli')
            info(f"Creating task cli[{name}] with {tg}")
            tg.create_task(cli.run(), name=f'cli[{name}]')

    async def create_session(self, context, sid=0):
        if sid in self._sessions:
            raise ValueError(f"Another session exists with the same {sid=}")
        info(f"Loading fuzzer config {self._argspace.config} with overrides")
        config = FuzzerConfig(self._argspace.config, self._overrides)
        info(f"Instantiating FuzzerSession with {config}")
        session = await config.instantiate('session', context, sid=sid)
        self._sessions[sid] = session
        return session

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

        def shell_exit(reader, writer):
            self._cleanup_and_exit(loop, 0)

        # Spawn a shell
        commands = {
            "Exit": (shell_exit, argparse.ArgumentParser(description="Exit Tango.")),
        }
        streams = get_standard_streams(use_stderr=True, loop=loop)
        self._cli = AsynchronousCli(commands, streams, prog="Tango")
        await self._cli.interact(stop=False, handle_sigint=False)

        self._suspendable.resume()

        if not self._cleanup:
            # reopen stdin if it is at EOF
            if self._cli.reader.at_eof():
                sys.stdin.close()
                sys.stdin = open("/dev/tty")

            self._timer()
            self._bootstrap_sigint(loop, handle=True)

    def _cleanup_and_exit(self, loop, code=0):
        self._cleanup = True
        self._runner.cancel()
        sys.exit(code)

    def run(self):
        # child watcher prints a warning when its children are reaped by
        # someone else. See FIXME in WebDataLoader
        logging.getLogger('asyncio').setLevel(logging.CRITICAL)
        info("Bootstraping and hopefully never returning :)")
        asyncio.run(self._bootstrap())
