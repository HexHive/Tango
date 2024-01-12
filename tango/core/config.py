from . import debug, info, warning, error

from tango.core.dataio  import AbstractChannelFactory
from tango.core.loader  import AbstractLoader
from tango.core.tracker  import AbstractTracker
from tango.core.explorer import AbstractExplorer
from tango.core.generator import AbstractInputGenerator
from tango.core.strategy import AbstractStrategy
from tango.core.session import FuzzerSession
from tango.common import (AsyncComponent, ComponentOwner,
    ComponentType, ComponentKey)

from functools import cached_property
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
        debug("Changed current working directory to %s", cwd)

        wd = self._config["fuzzer"]["work_dir"]
        resume = self._config["fuzzer"].get("resume", False)
        self.setup_workdir(wd, resume)
        debug(f"Setup workdir {wd} (resumed: {resume})")

        if "root" not in self._config["fuzzer"]:
            self._config["fuzzer"]["root"] = False
        root = self._config["fuzzer"]["root"]
        if root:
            ceuid = os.geteuid()
            os.setuid(0); os.seteuid(0)
            os.setgid(0); os.setegid(0)
            debug(f"Change current user {ceuid}(tango):{ceuid}(tango) to 0(root):0(root),{ceuid}(tango)")

        self.valid_component_classes = self.component_classes
        debug(f"Got {len(self.valid_component_classes)} valid component classes based on {file}")

    async def instantiate(self, component_type: ComponentKey, *args,
            config=None, **kwargs) -> AsyncComponent:
        component_type = ComponentType(component_type)
        config = config or self._config
        component = await self.valid_component_classes[component_type].instantiate(
                    self, config, *args, initialize=False, finalize=False,
                    **kwargs)
        strargs = (str(a) for a in args)
        strkwargs = ('='.join(str(x) for x in item) for item in kwargs.items())
        _args = ', '.join((*strargs, *strkwargs))
        debug('Created %s as %s(%s)', component_type, component.__class__.__name__, _args)

        # TODO: move to upper layer
        if not kwargs.get('dependants'):
            # initialization is breadth-first
            info(f"Initializing {component.__class__.__name__}'s dependencies")
            await component.initialize_dependencies(self, self.preinitialize_cb)
            # finalization is depth-first
            info(f"Finalizing {component.__class__.__name__}'s dependencies")
            await component.finalize_dependencies(self)
        return component

    def preinitialize_cb(self, component: AsyncComponent):
        object.__setattr__(component, '_entropy', self.entropy)

    def update_overrides(self, overrides: dict):
        def update(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        update(self._config, overrides)

    def setup_workdir(self, wd, resume):
        def mktree(root, tree):
            if tree:
                for b, t in tree.items():
                    mktree(os.path.join(root, b), t)
            else:
                Path(root).mkdir(parents=True, exist_ok=resume)
        tree = {
            "queue": {},
            "unstable": {},
            "crash": {},
            "figs": {},
            "shared": {}
        }
        mktree(wd, tree)

    @cached_property
    def entropy(self):
        seed = self._config["fuzzer"].get("entropy")
        return Random(seed)