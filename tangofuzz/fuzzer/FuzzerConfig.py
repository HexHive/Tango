# from functools    import cached_property
from common       import async_cached_property as cached_property
from loader       import Environment
from dataio    import (TCPChannelFactory,
                         TCPForkChannelFactory,
                         UDPChannelFactory,
                         UDPForkChannelFactory)
from loader       import ReplayStateLoader, ReplayForkStateLoader
from statemanager import StateManager
from statemanager.strategy import RandomStrategy, UniformStrategy
from tracker.coverage import CoverageStateTracker
from generator    import RandomInputGenerator
from random       import Random
import json
import os
import logging
import ctypes
from pathlib import Path
from subprocess import DEVNULL, PIPE

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
            "type": "<tcp | udp | raw | x11 | ...>",
            "tcp": {
                "endpoint": "address",
                "port": number,
                "connect_timeout": seconds,
                "data_timeout": seconds,
                "fork_location": "<listen | accept>"
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
            "type": "<mutation | generation | ...>",
            "spec": "/path/to/spec",
            "startup": "/path/to/pcap",
            ...
        },
        "statemanager": {
            "type": "<coverage | grammar | hybrid | ...>",
            "strategy": "<random | uniform | ...>",
            "validate_transitions": <true | false>,
            "minimize_transitions": <true | false>,
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
            so_path = os.path.join(self.lib_dir, "pytangofuzz.so")
            self._bind_lib = ctypes.CDLL(so_path)
        else:
            self._bind_lib = None

    @property
    def lib_dir(self):
        if (path := self._config["fuzzer"].get("lib")):
            return os.path.realpath(path)
        return None

    @cached_property
    async def exec_env(self):
        _config = self._config["exec"]
        for stdf in ["stdin", "stdout", "stderr"]:
            if stdf in _config:
                _config[stdf] = open(_config[stdf], "wt")
            elif stdf != "stdin":
                _config[stdf] = DEVNULL
            else:
                _config[stdf] = PIPE
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
    async def protocol(self):
        return self._config["channel"]["type"]

    @cached_property
    async def ch_env(self):
        _config = self._config["channel"]
        if not await self.use_forkserver:
            if _config["type"] == "tcp":
                return TCPChannelFactory(**_config["tcp"], \
                    timescale=await self.timescale)
            elif _config["type"] == "udp":
                return UDPChannelFactory(**_config["udp"], \
                    timescale=await self.timescale)
            else:
                raise NotImplemented()
        else:
            if _config["type"] == "tcp":
                fork_location = _config["tcp"].pop("fork_location", "accept")
                fork_before_accept = fork_location == "accept"
                return TCPForkChannelFactory(**_config["tcp"], \
                    timescale=await self.timescale, \
                    fork_before_accept=fork_before_accept)
            elif _config["type"] == "udp":
                return UDPForkChannelFactory(**_config["udp"], \
                    timescale=await self.timescale)
            else:
                raise NotImplemented()

    @cached_property
    async def loader(self):
        _config = self._config["loader"]
        if _config["type"] == "replay":
            if not await self.use_forkserver:
                kls = ReplayStateLoader
            else:
                kls = ReplayForkStateLoader
            return kls(
                ch_env=await self.ch_env,
                input_generator=await self.input_generator,
                exec_env=await self.exec_env,
                no_aslr=await self.disable_aslr
            )
        else:
            raise NotImplemented()

    @cached_property
    async def state_tracker(self):
        _config = self._config["statemanager"]
        state_type = _config.get("type", "coverage")
        loader = await self.loader

        if state_type == "coverage":
            tracker = await CoverageStateTracker.create(
                loader=loader,
                bind_lib=self._bind_lib
            )
        else:
            raise NotImplemented()

        # IMPORTANT must set the loader's state_tracker property to the tracker
        loader.state_tracker = tracker
        return tracker

    @cached_property
    async def input_generator(self):
        _config = self._config["input"]
        input_type = _config.get("type", "mutation")
        if input_type == "mutation":
            return RandomInputGenerator(await self.startup_pcap, await self.seed_dir, await self.protocol)
        else:
            raise NotImplemented()

    @cached_property
    async def entropy(self):
        seed = self._config["fuzzer"].get("entropy")
        return Random(seed)

    @cached_property
    async def scheduler_strategy(self):
        _config = self._config["statemanager"]
        strategy_name = _config.get("strategy", "random")
        entropy = await self.entropy
        if strategy_name == "random":
            return lambda a, b: RandomStrategy(sm=a, entry_state=b, entropy=entropy)
        elif strategy_name == "uniform":
            return lambda a, b: UniformStrategy(sm=a, entry_state=b, entropy=entropy)
        else:
            raise NotImplemented()

    @cached_property
    async def state_manager(self):
        _config = self._config["statemanager"]
        validate = _config.get("validate_transitions", True)
        minimize = _config.get("minimize_transitions", True)
        return StateManager(await self.loader, await self.state_tracker,
            await self.scheduler_strategy, await self.work_dir, await self.protocol,
            validate, minimize)

    @cached_property
    async def startup_pcap(self):
        return self._config["input"].get("startup")

    @cached_property
    async def seed_dir(self):
        return self._config["fuzzer"].get("seeds")

    @cached_property
    async def work_dir(self):
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
    async def resume(self):
        return self._config["fuzzer"].get("resume", False)

    @cached_property
    async def timescale(self):
        return self._config["fuzzer"].get("timescale", 1.0)

    @cached_property
    async def use_forkserver(self):
        return self._config["loader"].get("forkserver", False)

    @cached_property
    async def disable_aslr(self):
        return self._config["loader"].get("disable_aslr", False)
