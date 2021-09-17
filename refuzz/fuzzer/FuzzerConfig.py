from functools    import cached_property
from loader       import Environment
from networkio    import (TCPChannelFactory,
                         UDPChannelFactory)
from loader       import ReplayStateLoader
from statemanager import (CoverageStateTracker,
                         StateManager)
from generator    import RandomInputGenerator
from random       import Random
import json
import os

class FuzzerConfig:
    """
    A parser for the JSON-formatted fuzzer configuration file.

    The file has the following structure:
    {
        "exec": {
            "path": "/path/to/target",
            "args": ["/path/to/target", "-arg1", "value", "-arg2", "..."],
            "env": {
                "var1": "value1",
                "var2": "value2"
            },
            "cwd": "/path/to/working/dir",
            "stdin": "/path/to/input.txt",
            "stdout": "/path/to/output.log",
            "stderr": "/path/to/error.log",
            ...
        },
        "channel": {
            "type": "<tcp | udp | raw | ...>",
            "tcp": {
                "endpoint": "address",
                "port": number,
                "connect_timeout": seconds,
                "data_timeout": seconds
            },
            "udp": {
                "endpoint": "address",
                "port": number
            },
            ...
        },
        "loader": {
            "type": "<replay | snapshot | ...>",
            ...
        },
        "input": {
            "type": "<mutation | generation  | ...>",
            "spec": "/path/to/spec",
            "startup": "/path/to/pcap",
            ...
        },
        "statemanager": {
            "type": "<coverage | grammar | hybrid | ...>",
            "strategy": "<depth-first | breadth-first | ...>",
            ...
        },
        "fuzzer": {
            "seeds": "/path/to/pcap/dir",
            "timescale": .0 .. 1.,
            "entropy": number,
            ...
        }
    }
    """
    def __init__(self, file: str):
        """
        Reads and parses a JSON-formatted configuration file.

        :param      file:  The path to the JSON file
        :type       file:  str
        """
        with open(file, "rt") as f:
            self._config = json.load(f)

    @cached_property
    def exec_env(self):
        _config = self._config["exec"]
        for stdf in ["stdin", "stdout", "stderr"]:
            if stdf in _config:
                _config[stdf] = open(_config[stdf], "wt")
        if not _config.get("env"):
            _config["env"] = dict(os.environ)
        return Environment(**_config)

    @cached_property
    def ch_env(self):
        _config = self._config["channel"]
        if _config["type"] == "tcp":
            return TCPChannelFactory(**_config["tcp"], timescale=self.timescale)
        elif _config["type"] == "udp":
            return UDPChannelFactory(**_config["udp"], timescale=self.timescale)
        else:
            raise NotImplemented()

    @cached_property
    def loader(self):
        _config = self._config["loader"]
        if _config["type"] == "replay":
            return ReplayStateLoader(self.exec_env, self.ch_env)
        else:
            raise NotImplemented()

    @cached_property
    def state_tracker(self):
        _config = self._config["statemanager"]
        state_type = _config.get("type", "coverage")
        if state_type == "coverage":
            return CoverageStateTracker(self.input_generator, self.loader)
        else:
            raise NotImplemented()

    @cached_property
    def input_generator(self):
        _config = self._config["input"]
        input_type = _config.get("type", "mutation")
        if input_type == "mutation":
            return RandomInputGenerator(self.startup_pcap, self.seed_dir, self.ch_env)
        else:
            raise NotImplemented()

    @cached_property
    def entropy(self):
        seed = self._config["fuzzer"].get("entropy")
        return Random(seed)

    @cached_property
    def scheduler_strategy(self):
        return self._config["statemanager"].get("strategy")

    @cached_property
    def state_manager(self):
        return StateManager(self.input_generator.startup_input,
            self.loader, self.state_tracker, self.scheduler_strategy)

    @cached_property
    def startup_pcap(self):
        return self._config["input"].get("startup")

    @cached_property
    def seed_dir(self):
        return self._config["fuzzer"].get("seeds")

    @cached_property
    def timescale(self):
        return self._config["fuzzer"].get("timescale", 1.0)
