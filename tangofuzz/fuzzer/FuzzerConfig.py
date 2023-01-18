# from functools    import cached_property
from . import debug, info
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
from generator    import RandomInputGenerator, ReactiveInputGenerator, StatelessReactiveInputGenerator
from random       import Random
import json
import os
import logging
import ctypes
from pathlib import Path
from subprocess import DEVNULL, PIPE
from profiler import ProfileValue
import collections.abc

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
                "fork_location": "<listen | accepting>"
            },
            "udp": {
                "endpoint": "address",
                "port": number,
                "fork_location": "<bind | binding>"
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
            "type": "<random | reactive | reactless | ...>",
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
            "cwd": "/path/to/cwd",
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
    def __init__(self, file: str, overrides: dict=None):
        """
        Reads and parses a JSON-formatted configuration file.

        :param      file:  The path to the JSON file
        :type       file:  str

        :param      overrides: A nested dict of keys to override those in the
                               main config.
        :type       overrides: dict
        """
        with open(file, "rt") as f:
            self._config = json.load(f)

        if overrides:
            self.update_overrides(overrides)

        if self.lib_dir:
            so_path = os.path.join(self.lib_dir, "pytangofuzz.so")
            self._bind_lib = ctypes.CDLL(so_path)
        else:
            self._bind_lib = None

        os.chdir(self.cwd)
        info(f'Changed current working directory to {self.cwd}')

    @property
    def lib_dir(self):
        if (path := self._config["fuzzer"].get("lib")):
            return os.path.realpath(path)
        return None

    @property
    def cwd(self):
        cwd = self._config["fuzzer"].get("cwd", ".")
        return cwd

    def update_overrides(self, overrides: dict):
        def update(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        update(self._config, overrides)

    @cached_property
    async def exec_env(self):
        _config = self._config["exec"]
        ProfileValue('target_name')(_config["path"])
        for stdf in ["stdin", "stdout", "stderr"]:
            if _config.get(stdf) == "inherit":
                _config[stdf] = None
            elif _config.get(stdf) is not None:
                _config[stdf] = open(_config[stdf], "wt")
            elif stdf == "stdin":
                _config[stdf] = PIPE
            else:
                _config[stdf] = DEVNULL
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
                raise NotImplementedError()
        else:
            if _config["type"] == "tcp":
                fork_location = _config["tcp"].pop("fork_location", "accepting")
                fork_before_accept = fork_location == "accepting"
                return TCPForkChannelFactory(**_config["tcp"], \
                    timescale=await self.timescale, \
                    fork_before_accept=fork_before_accept)
            elif _config["type"] == "udp":
                fork_location = _config["udp"].pop("fork_location", "bind")
                fork_before_bind = fork_location == "binding"
                return UDPForkChannelFactory(**_config["udp"], \
                    timescale=await self.timescale, \
                    fork_before_bind=fork_before_bind)
            else:
                raise NotImplementedError()

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
            raise NotImplementedError()

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
            raise NotImplementedError()

        # IMPORTANT must set the loader's state_tracker property to the tracker
        loader.state_tracker = tracker
        return tracker

    @cached_property
    async def input_generator(self):
        _config = self._config["input"]
        input_type = _config.get("type", "random")
        if input_type == "random":
            return RandomInputGenerator(await self.startup_pcap, await self.seed_dir, await self.protocol)
        elif input_type == "reactive":
            return ReactiveInputGenerator(await self.startup_pcap, await self.seed_dir, await self.protocol, await self.work_dir)
        elif input_type == "reactless":
            return StatelessReactiveInputGenerator(await self.startup_pcap, await self.seed_dir, await self.protocol, await self.work_dir)
        else:
            raise NotImplementedError()

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
            raise NotImplementedError()

    @cached_property
    async def state_manager(self):
        _config = self._config["statemanager"]
        validate = _config.get("validate_transitions", True)
        minimize = _config.get("minimize_transitions", True)
        return StateManager(await self.input_generator, await self.loader,
            await self.state_tracker, await self.scheduler_strategy,
            await self.work_dir, await self.protocol,
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
