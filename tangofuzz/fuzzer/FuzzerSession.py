from . import debug, info, warning, critical, error

from input         import (InputBase,
                          PreparedInput,
                          FileCachingDecorator)
from fuzzer        import FuzzerConfig
from common        import (StabilityException,
                          StatePrecisionException,
                          LoadedException,
                          ChannelTimeoutException,
                          ChannelBrokenException,
                          ChannelSetupException,
                          ProcessCrashedException,
                          ProcessTerminatedException,
                          StateNotReproducibleException)
import os

from profiler import (ProfileLambda,
                     ProfileCount,
                     ProfiledObjects,
                     ProfilingStoppedEvent,
                     ProfileTimeElapsed)

from webui import WebRenderer
import asyncio

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

    @classmethod
    async def create(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
        self._input_gen = await self._config.input_generator
        self._loader = await self._config.loader
        self._sman = await self._config.state_manager
        self._entropy = await self._config.entropy
        self._workdir = await self._config.work_dir
        self._protocol = await self._config.protocol

        ## After this point, the StateManager and StateTracker are both
        #  initialized and should be able to identify states and populate the SM
        return self

    async def _load_seeds(self):
        """
        Loops over the initial set of seeds to populate the state machine with
        known states.
        """
        for input in self._input_gen.seeds:
            try:
                await self._sman.reset_state()
                # feed input to target and populate state machine
                await self._loader.execute_input(input, self._sman)
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
                            # TODO save crashing input
                            error(f"Process crashed: {pc = }")
                            ProfileCount('crash')(1)
                            FileCachingDecorator(self._workdir, "crash", self._protocol)(ex.payload, self._sman, copy=True)
                        except ProcessTerminatedException as pt:
                            debug(f"Process terminated unexpectedtly? ({pt = })")
                        except ChannelTimeoutException:
                            # TODO save timeout input
                            warning("Received channel timeout exception")
                            ProfileCount('timeout')(1)
                        except ChannelBrokenException as ex:
                            # TODO save crashing/breaking input
                            debug(f"Received channel broken exception ({ex = })")
                        except StateNotReproducibleException as ex:
                            warning(f"Target state {ex._faulty_state} not reachable anymore!")
                        except ChannelSetupException:
                            # TODO save broken setup input
                            warning("Received channel setup exception")
                        except Exception as ex:
                            import ipdb; ipdb.set_trace()
                            critical(f"Encountered unhandled loaded exception {ex = }")
                        # reset to StateManager's current target state
                        await self._sman.reload_target()
                    except asyncio.CancelledError:
                        # the input generator cancelled an execution and would
                        # like to react
                        warning("Received interrupt, continuing!")
                        continue
                    except Exception as ex:
                        import ipdb; ipdb.set_trace()
                        critical(f"Encountered weird exception {ex = }")
                        await self._sman.reload_target()
            except asyncio.CancelledError:
                # the input generator cancelled an execution and would like to
                # react
                while True:
                    try:
                        warning("Received interrupt while reloading target, forcing input generation")
                        cur_state = self._sman.state_tracker.current_state
                        input = self._input_gen.generate(cur_state, self._entropy)
                        await self._sman.step(input)
                        break
                    except asyncio.CancelledError:
                        continue
                continue
            except StateNotReproducibleException as ex:
                warning(f"Target state {ex._faulty_state} not reachable anymore!")
            except Exception as ex:
                critical(f"Encountered exception while resetting state! {ex = }")
            except KeyboardInterrupt:
                from code import InteractiveConsole
                # it seems this library fixes the handling of arrows and other
                # special console signals
                import readline
                repl = InteractiveConsole(locals=locals())
                ProfiledObjects['elapsed'].toggle()
                repl.interact(banner="Fuzzing paused (type exit() to quit)",
                    exitmsg="Fuzzing resumed")
                ProfiledObjects['elapsed'].toggle()

    async def start(self):
        await self._load_seeds()

        # reset state after the seed initialization stage
        await self._sman.reset_state()

        ProfileTimeElapsed('elapsed')

        # FIXME the WebRenderer is async and can thus be started in the same loop
        WebRenderer(self).start()

        # launch fuzzing loop
        await self._loop()

        # TODO anything else?
