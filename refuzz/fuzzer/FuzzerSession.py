from input         import (InputBase,
                          PreparedInput)
from fuzzer        import FuzzerConfig
import os

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

        self._loader = config.loader
        self._sman = config.state_manager
        self._seed_dir = config.seed_dir
        self._ch_env = config.ch_env # we may need this for parsing PCAP

        self._load_seeds()

    def _load_seeds(self):
        """
        Loops over the initial set of seeds to populate the state machine with
        known states.
        """
        if self._seed_dir is None or not os.path.isdir(self._seed_dir):
            return

        seeds = []
        for root, _, files in os.walk(self._seed_dir):
            seeds.extend(os.path.join(root, file) for file in files)

        for seed in seeds:
            # parse seed to PreparedInput
            input = PCAPInput(seed, self._ch_env)

            # restore to entry state
            self._sman.reset_state()
            # feed input to target and populate state machine
            self._loader.execute_input(input, self._loader.channel, self._sman)

    def _loop(self):
        # FIXME is there ever a proper terminating condition for fuzzing?
        while True:
            cur_state = self._sman.state_tracker.current_state
            input = cur_state.get_escaper()
            self._loader.execute_input(input, self._loader.channel, self._sman)

    def start(self):
        # reset state after the seed initialization stage
        self._sman.reset_state()

        # launch fuzzing loop
        self._loop()

        # TODO anything else?
        pass