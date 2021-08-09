from input         import (InputBase,
                          PreparedInput)
from fuzzer        import FuzzerConfig
import os
from threading import Timer
from time import sleep

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

        self._input_gen = config.input_generator
        self._loader = config.loader
        self._sman = config.state_manager
        self._entropy = config.entropy

        ## After this point, the StateManager and StateTracker are both
        #  initialized and should be able to identify states and populate the SM

        # stats stuff
        self._counter = 0

        self._load_seeds()

    def _load_seeds(self):
        """
        Loops over the initial set of seeds to populate the state machine with
        known states.
        """
        for input in self._input_gen.seeds:
            self._sman.reset_state()
            # feed input to target and populate state machine
            self._loader.execute_input(input, self._sman)

    def _loop(self):
        # FIXME is there ever a proper terminating condition for fuzzing?
        while True:
            cur_state = self._sman.state_tracker.current_state
            input = self._input_gen.generate(cur_state, self._entropy)
            self._loader.execute_input(input, self._sman)
            self._sman.step()
            self._counter += 1

    def _stats(self):
        while True:
            print(f"execs/s {self._counter / 5.0}")
            self._counter = 0
            sleep(5)

    def start(self):
        # reset state after the seed initialization stage
        self._sman.reset_state()

        Timer(5.0, self._stats).start()

        # launch fuzzing loop
        self._loop()

        # TODO anything else?
        pass