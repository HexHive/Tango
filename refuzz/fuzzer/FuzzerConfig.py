from functools    import cached_property
from loader       import Environment
from networkio    import (TCPChannelFactory,
                         TCPForkChannelFactory,
                         UDPChannelFactory)
from loader       import ReplayStateLoader, ReplayForkStateLoader
from statemanager import (CoverageStateTracker,
                         StateManager,
                         RandomStrategy)
from generator    import RandomInputGenerator
from random       import Random
import json
import os
import logging
import ctypes
from pathlib import Path

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
            "forkserver": <true | false>,
            "disable_aslr": <true | false>,
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
            "strategy": "<random | ...>",
            "cache_inputs": <true | false>,
            ...
        },
        "fuzzer": {
            "workdir": "/path/to/workdir",
            "resume": <true | false>,
            "seeds": "/path/to/pcap/dir",
            "timescale": .0 .. 1.,
            "entropy": number,
            "lib": "/path/to/tangofuzz/lib",
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

        if self.lib_dir:
            so_path = os.path.join(self.lib_dir, "rebind.so")
            self._bind_lib = ctypes.CDLL(so_path)
        else:
            self._bind_lib = None

    @cached_property
    def exec_env(self):
        _config = self._config["exec"]
        for stdf in ["stdin", "stdout", "stderr"]:
            if stdf in _config:
                _config[stdf] = open(_config[stdf], "wt")
        if not _config.get("env"):
            _config["env"] = dict(os.environ)
        _config["args"][0] = os.path.realpath(_config["args"][0])
        if not (path := _config.get("path")):
            _config["path"] = _config["args"][0]
        else:
            _config["path"] = os.path.realpath(path)
        if (cwd := _config.get("cwd")):
            _config["cwd"] = os.path.realpath(cwd)
        return Environment(**_config)

    @cached_property
    def ch_env(self):
        _config = self._config["channel"]
        if not self.use_forkserver:
            if _config["type"] == "tcp":
                return TCPChannelFactory(**_config["tcp"], timescale=self.timescale)
            else:
                raise NotImplemented()
        else:
            if _config["type"] == "tcp":
                return TCPForkChannelFactory(**_config["tcp"], timescale=self.timescale)
            else:
                raise NotImplemented()

    @cached_property
    def loader(self):
        _config = self._config["loader"]
        if _config["type"] == "replay":
            if not self.use_forkserver:
                return ReplayStateLoader(self.exec_env, self.ch_env, self.disable_aslr)
            else:
                return ReplayForkStateLoader(self.exec_env, self.ch_env, self.disable_aslr)
        else:
            raise NotImplemented()

    @cached_property
    def state_tracker(self):
        _config = self._config["statemanager"]
        state_type = _config.get("type", "coverage")
        if state_type == "coverage":
            return CoverageStateTracker(self.input_generator, self.loader,
                bind_lib=self._bind_lib)
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
        _config = self._config["statemanager"]
        strategy_name = _config.get("strategy", "random")
        if strategy_name == "random":
            return lambda sm, startup: RandomStrategy(sm, startup, entropy=self.entropy)
        else:
            raise NotImplemented()

    @cached_property
    def state_manager(self):
        _config = self._config["statemanager"]
        cache_inputs = _config.get("cache_inputs", True)
        return StateManager(self.input_generator.startup_input,
            self.loader, self.state_tracker, self.scheduler_strategy,
            cache_inputs, self.work_dir, self.ch_env.protocol)

    @cached_property
    def startup_pcap(self):
        return self._config["input"].get("startup")

    @cached_property
    def seed_dir(self):
        return self._config["fuzzer"].get("seeds")

    @cached_property
    def lib_dir(self):
        if (path := self._config["fuzzer"].get("lib")):
            return os.path.realpath(path)
        return None

    @cached_property
    def work_dir(self):
        def mktree(root, tree):
            if tree:
                for b, t in tree.items():
                    mktree(os.path.join(root, b), t)
            else:
                Path(root).mkdir(parents=True, exist_ok=self.resume)

        wd = self._config["fuzzer"]["workdir"]
        tree = {
            "queue": {},
            "crash": {}
        }
        mktree(wd, tree)
        return wd

    @cached_property
    def resume(self):
        return self._config["fuzzer"].get("resume", False)

    @cached_property
    def timescale(self):
        return self._config["fuzzer"].get("timescale", 1.0)

    @cached_property
    def use_forkserver(self):
        return self._config["loader"].get("forkserver", False)

    @cached_property
    def disable_aslr(self):
        return self._config["loader"].get("disable_aslr", False)