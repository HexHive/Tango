from . import debug, info, warning, critical, error

from input         import (InputBase,
                          PreparedInput)
from fuzzer        import FuzzerConfig
from common        import (StabilityException,
                          StatePrecisionException,
                          LoadedException,
                          ChannelTimeoutException,
                          ChannelBrokenException,
                          ChannelSetupException,
                          ProcessCrashedException,
                          ProcessTerminatedException,
                          StateNotReproducibleException,
                          CoroInterrupt)
from common        import Suspendable
import os
import sys

import profiler
from profiler import (ProfileLambda,
                     ProfileCount,
                     ProfiledObjects,
                     ProfileTimeElapsed,
                     ProfilingTasks)

from webui import WebRenderer
import asyncio
from aioconsole import AsynchronousConsole, get_standard_streams
import signal
import logging

class FuzzerSession:
    """
    This class initializes and tracks the global state of the fuzzer.
    """
    def __init__(self, config: FuzzerConfig):
        """
        FuzzerSession initializer.
        Takes the fuzzer configuration, initial seeds, and other parameters as
        arguments.
        Initializes the target and establishes a communication channel.

        :param      config: The fuzzer configuration object
        :type       config: FuzzerConfig
        """
        self._config = config

    async def initialize(self):
        self._input_gen = await self._config.input_generator
        self._loader = await self._config.loader
        self._sman = await self._config.state_manager
        self._entropy = await self._config.entropy
        self._workdir = await self._config.work_dir

        ## After this point, the StateManager and StateTracker are both
        #  initialized and should be able to identify states and populate the SM

    async def _load_seeds(self):
        """
        Loops over the initial set of seeds to populate the state machine with
        known states.
        """
        # TODO also load in existing queue if config.resume is True
        for input in self._input_gen.seeds:
            try:
                await self._sman.reset_state()
                # feed input to target and populate state machine
                context_input = self._sman.get_context_input(input)
                await self._loader.execute_input(context_input)
                info(f"Loaded seed file: {input}")
            except LoadedException as ex:
                warning(f"Failed to load {input}: {ex.exception}")

    async def _loop(self):
        # FIXME is there ever a proper terminating condition for fuzzing?
        while True:
            try:
                # reset to StateManager's current target state
                # FIXME this should probably be done by the exploration strategy
                await self._sman.reload_target()
                while True:
                    try:
                        cur_state = self._sman.state_tracker.current_state
                        input = self._input_gen.generate(cur_state, self._entropy)
                        await self._sman.step(input)
                    except LoadedException as ex:
                        try:
                            raise ex.exception
                        except StabilityException:
                            ProfileCount('unstable')(1)
                        except StatePrecisionException:
                            debug("Encountered imprecise state transition")
                            ProfileCount('imprecise')(1)
                        except ProcessCrashedException as pc:
                            error(f"Process crashed: {pc = }")
                            ProfileCount('crash')(1)
                            # TODO augment loader to dump stdout and stderr too
                            self._input_gen.save_input(ex.payload, self._sman._current_path, 'crash', repr(self._sman._current_state))
                        except ProcessTerminatedException as pt:
                            debug(f"Process terminated unexpectedtly? ({pt = })")
                        except ChannelTimeoutException:
                            # TODO save timeout input
                            warning("Received channel timeout exception")
                            ProfileCount('timeout')(1)
                        except ChannelBrokenException as ex:
                            # TODO save crashing/breaking input
                            debug(f"Received channel broken exception ({ex = })")
                        except ChannelSetupException:
                            # TODO save broken setup input
                            warning("Received channel setup exception")
                        except (CoroInterrupt, StateNotReproducibleException, asyncio.CancelledError):
                            # we pass these over to the next try-catch
                            raise
                        except Exception as ex:
                            # everything else, we probably need to debug
                            critical(f"Encountered unhandled loaded exception {ex = }")
                            import ipdb; ipdb.set_trace()
                            raise
                        # reset to StateManager's current target state
                        await self._sman.reload_target()
                    except CoroInterrupt:
                        # the input generator cancelled an execution
                        warning("Received interrupt, continuing!")
                        continue
                    except (StateNotReproducibleException, asyncio.CancelledError):
                        # we pass these over to the next try-catch
                        raise
                    except Exception as ex:
                        # everything else, we probably need to debug
                        critical(f"Encountered weird exception {ex = }")
                        import ipdb; ipdb.set_trace()
                        raise
            except CoroInterrupt:
                # the input generator cancelled an execution
                warning("Received interrupt while reloading target")
                continue
            except StateNotReproducibleException as ex:
                warning(f"Target state {ex._faulty_state} not reachable anymore!")
            except asyncio.CancelledError:
                return
            except Exception as ex:
                import ipdb; ipdb.set_trace()
                critical(f"Encountered exception while resetting state! {ex = }")

    async def _start(self):
        await self._load_seeds()

        # reset state after the seed initialization stage
        await self._sman.reset_state()

        ProfileTimeElapsed('elapsed')

        # launch fuzzing loop
        await self._loop()

        # TODO anything else?

    async def _bootstrap(self):
        loop = asyncio.get_running_loop()
        self._stream_cache = {}
        # we use an empty tuple for `streams` to prevent it being seen as None
        self._repl = AsynchronousConsole(streams=(), loop=loop,
            locals=locals() | {
                'exit': lambda code=0: self._cleanup_and_exit(loop, code),
                'loop': loop
            })

        await self.initialize()
        await profiler.initialize()

        # start WebUI
        webui_task = loop.create_task(WebRenderer(self).run())
        ProfilingTasks.append(webui_task)

        # set up the main suspendable fuzzing task
        suspendable = Suspendable(self._start())
        loop.main_task = main_task = loop.create_task(suspendable.as_coroutine())
        main_task.suspendable = suspendable

        self._bootstrap_sigint(loop, handle=True)

        try:
            await asyncio.gather(main_task, *ProfilingTasks)
        except asyncio.CancelledError:
            pass
        except Exception as ex:
            warning(f"Exception raised while attempting to terminate: {ex}")
        info("Goodbye!")

    def _bootstrap_sigint(self, loop, handle=True):
        if handle:
            loop.add_signal_handler(signal.SIGINT, self._sigint_handler, loop)
        else:
            loop.remove_signal_handler(signal.SIGINT)

    def _sigint_handler(self, loop):
        async def await_repl(task, restore):
            await task
            loop.main_task.suspendable.set_tasks(wakeup_task=restore)
        repl_task = loop.create_task(self._interact(loop))
        _, wakeup_restore = loop.main_task.suspendable.tasks
        loop.main_task.suspendable.set_tasks(wakeup_task=await_repl(repl_task, wakeup_restore))

    async def _interact(self, loop):
        main_task = loop.main_task
        assert main_task != asyncio.current_task()

        # disable handler
        self._bootstrap_sigint(loop, handle=False)

        # suspend main task (and call respective suspend handlers)
        main_task.suspendable.suspend()

        # pause fuzzing timer
        ProfiledObjects['elapsed'].toggle()

        # The coroutine get_standard_streams is deliberately not awaited.
        # AsynchronousConsole supports having an awaitable `streams` attribute
        # that is called every time `interact` is invoked. To allow for reusing
        # the console, we use this coroutine with a no-op cache so that streams
        # are re-generated every time `interact` is called.
        # interact.
        # Unfortunately, AsynchronousConsole sets `self.streams` to None after
        # evaluating it, so we need to re-assign it.
        self._stream_cache.clear()
        self._repl.streams = get_standard_streams(cache=self._stream_cache, use_stderr=True, loop=loop)
        await self._repl.interact(banner="Fuzzing paused (type exit() to quit)",
            stop=False, handle_sigint=False)

        # reopen stdin if it is at EOF
        if self._repl.reader.at_eof():
            sys.stdin.close()
            sys.stdin = open("/dev/tty")

        # reverse setup
        ProfiledObjects['elapsed'].toggle()
        main_task.suspendable.resume()
        self._bootstrap_sigint(loop, handle=True)

    def _cleanup_and_exit(self, loop, code, /):
        profiler.ProfilingStoppedEvent.set()
        loop.main_task.cancel()
        # for task in asyncio.all_tasks():
        #     task.cancel()
        sys.exit(code)

    def run(self):
        # child watcher prints a warning when its children are reaped by
        # someone else. See FIXME in WebDataLoader
        logging.getLogger('asyncio').setLevel(logging.CRITICAL)
        asyncio.run(self._bootstrap())
