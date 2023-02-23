from . import  info

from tango.core.dataio  import AbstractChannelFactory
from tango.core.loader  import AbstractStateLoader
from tango.core.tracker  import AbstractStateTracker
from tango.core.explorer import AbstractExplorer
from tango.core.generator import AbstractInputGenerator
from tango.core.strategy import AbstractStrategy
from tango.core.session import FuzzerSession
from tango.common       import (async_cached_property, cached_property,
    AsyncComponent, ComponentOwner, ComponentType, ComponentKey)

from random import Random
from pathlib import Path
import json
import os
import collections.abc

__all__ = ['FuzzerConfig']

class FuzzerConfig(ComponentOwner):
    """
    A parser for the JSON-formatted fuzzer configuration file.
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
        super().__init__(self._config)

        cwd = self._config["fuzzer"].get("cwd", ".")
        os.chdir(cwd)
        info(f'Changed current working directory to {cwd}')
        self.setup_workdir()

    async def instantiate(self, component_type: ComponentKey, config=None,
            *args, **kwargs) -> AsyncComponent:
        factory = None
        component_type = ComponentType(component_type)
        if config is not None:
            # was not called through own async_cached_property
            lookup_attr = component_type.name
            if hasattr(self, lookup_attr):
                factory = getattr(self, lookup_attr)
        if not factory:
            config = config or self._config
            factory = self.component_classes[component_type].instantiate( \
                        self, config, *args, initialize=False, **kwargs)
        component = await factory
        if not component._initialized:
            component._entropy = self.entropy
            await component.initialize()
            component._initialized = True
        return component

    def update_overrides(self, overrides: dict):
        def update(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        update(self._config, overrides)

    def setup_workdir(self):
        resume = self._config["fuzzer"].get("resume", False)
        def mktree(root, tree):
            if tree:
                for b, t in tree.items():
                    mktree(os.path.join(root, b), t)
            else:
                Path(root).mkdir(parents=True, exist_ok=resume)

        wd = self._config["fuzzer"]["work_dir"]
        tree = {
            "queue": {},
            "unstable": {},
            "crash": {},
            "figs": {}
        }
        mktree(wd, tree)

    @cached_property
    def entropy(self):
        seed = self._config["fuzzer"].get("entropy")
        return Random(seed)

    ### AsyncComponent components ###

    @async_cached_property
    async def channel_factory(self) -> AbstractChannelFactory:
        return await self.instantiate(ComponentType.channel_factory)

    @async_cached_property
    async def loader(self) -> AbstractStateLoader:
        return await self.instantiate(ComponentType.loader)

    @async_cached_property
    async def tracker(self) -> AbstractStateTracker:
        return await self.instantiate(ComponentType.tracker)

    @async_cached_property
    async def explorer(self) -> AbstractExplorer:
        return await self.instantiate(ComponentType.explorer)

    @async_cached_property
    async def generator(self) -> AbstractInputGenerator:
        fmt = (await self.channel_factory).fmt
        return await self.instantiate(ComponentType.generator, fmt=fmt)

    @async_cached_property
    async def strategy(self) -> AbstractStrategy:
        return await self.instantiate(ComponentType.strategy)

    @async_cached_property
    async def session(self) -> FuzzerSession:
        return await self.instantiate(ComponentType.session)