import json
from functools import cached_property
from loader import (Environment,
                   TCPChannelFactory,
                   UDPChannelFactory,
                   ReplayStateLoader)
from statemanager import (CoverageStateTracker,
                         StateManager)

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
        "fuzzer": {
            "type": "<coverage | grammar | hybrid | ...>",
            "strategy": "<depth-first | breadth-first | ...>",
            "seeds": "/path/to/pcap/dir",
            "timescale": .0 .. 1.,
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
        _config = self._config["fuzzer"]
        if _config["type"] == "coverage":
            return CoverageStateTracker(self.loader)
        else:
            raise NotImplemented()

    @cached_property
    def scheduler_strategy(self):
        return self._config["fuzzer"]["strategy"]

    @cached_property
    def state_manager(self):
        return StateManager(self.loader, self.state_tracker,
                self.scheduler_strategy)

    @cached_property
    def seed_dir(self):
        return self._config["fuzzer"].get("seeds")

    @cached_property
    def timescale(self):
        return self._config["fuzzer"].get("timescale", 1.0)
