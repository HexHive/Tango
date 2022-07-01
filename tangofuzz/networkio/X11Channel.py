from __future__ import annotations
from . import debug
from networkio import ChannelBase, ChannelFactoryBase
from common import ChannelSetupException, sync_to_async, async_property, async_suspendable
from typing import Callable

from dataclasses import dataclass
from Xlib import XK, display, ext, X, protocol
import sys
import time
from multiprocessing import Lock
from statistics import mean, stdev
from collections import deque

@dataclass
class X11ChannelFactory(ChannelFactoryBase):
    struct_fn: Callable
    protocol: str = 'x11'

    def create(self, program_name=None, window_name=None, *args, **kwargs) -> X11Channel:
        return X11Channel(self.struct_fn, program_name=program_name, window_name=window_name)

class X11Channel(ChannelBase):
    WINDOW_POLL_RETRIES = 10
    WINDOW_POLL_WAIT = 1.0

    def __init__(self, struct_fn, *, program_name, window_name, **kwargs):
        super().__init__(**kwargs)
        self._display = display.Display()
        self._struct_fn = struct_fn

        retries = 0
        while True:
            try:
                self._window = self.get_window(self._display, program_name, window_name)
            except ChannelSetupException:
                retries += 1
                if retries == self.WINDOW_POLL_RETRIES:
                    raise
                time.sleep(self.WINDOW_POLL_WAIT)
            else:
                break
        self._keysdown = set()

        self._suspend_called = 0
        self._mutex = Lock()
        self._samples = deque(maxlen=100)

    @async_property
    async def timescale(self):
        tic_rate = (await self._struct_fn()).tic_rate
        self._samples.append(tic_rate)
        tic_rate = mean(self._samples)
        tic_std = stdev(self._samples) if len(self._samples) > 1 else 0.0
        # debug(f'mean {tic_rate=} stdev={tic_std}')
        return 35 / tic_rate if tic_rate >= 35 else 1

    @classmethod
    def get_window(cls, display, program_name, window_name):
        root_win = display.screen().root
        window_list = [root_win]

        while len(window_list) != 0:
            win = window_list.pop(0)
            if program_name is None \
                    or (win.get_wm_class() or ('',))[0] == program_name:
                if window_name is None or win.get_wm_name() == window_name:
                    return win.id
            children = win.query_tree().children
            if children != None:
                window_list += children

        raise ChannelSetupException("Failed to find x11 window")

    async def send(self, key_or_keys, down=True, clobbers=None):
        if hasattr(key_or_keys, '__iter__') and not isinstance(key_or_keys, str):
            keys = key_or_keys
        else:
            keys = (key_or_keys,)
        clobbers = clobbers or ()

        if down:
            for c in clobbers:
                if c in self._keysdown:
                    await self._send_one_key_event_safe(c, down=False)

        for k in keys:
            await self._send_one_key_event_safe(k, down)

        # debug(self._keysdown)

    def sync_send(self, key_or_keys, down=True, clobbers=None):
        if hasattr(key_or_keys, '__iter__') and not isinstance(key_or_keys, str):
            keys = key_or_keys
        else:
            keys = (key_or_keys,)
        clobbers = clobbers or ()

        if down:
            for c in clobbers:
                if c in self._keysdown:
                    self._sync_send_one_key_event(c, down=False)

        for k in keys:
            self._sync_send_one_key_event(k, down)

        # debug(self._keysdown)

    async def _send_one_key_event_safe(self, key, down):
        return await self._send_one_key_event(key, down)

    @async_suspendable(suspend_cb='_send_one_key_suspend_callback',
                       resume_cb='_send_one_key_resume_callback')
    @sync_to_async(done_cb='_send_one_key_done_callback')
    def _send_one_key_event(self, key, down):
        return self._sync_send_one_key_event(key, down)

    def _sync_send_one_key_event(self, key, down):
        with self._mutex:
            keysym = XK.string_to_keysym(key)
            keycode = self._display.keysym_to_keycode(keysym)

            event_ctor = protocol.event.KeyPress if down else protocol.event.KeyRelease

            event = event_ctor(
               time = int(time.time()),
               root = self._display.screen().root,
               window = self._window,
               same_screen = 0, child = X.NONE,
               root_x = 0, root_y = 0, event_x = 0, event_y = 0,
               state = 0,
               detail = keycode
            )
            self._display.send_event(self._window, event, propagate=True)
            self._display.sync()

            if down:
                self._keysdown.add(key)
            else:
                self._keysdown.discard(key)

    def _send_one_key_done_callback(self, key, down, *, future):
        if self._suspend_called and down:
            self._sync_send_one_key_event(key, False)

    def _send_one_key_suspend_callback(self, key, down):
        self._suspend_called += 1

    def _send_one_key_resume_callback(self, key, down):
        if self._suspend_called and down:
            self._suspend_called -= 1
            self._sync_send_one_key_event(key, True)

    async def receive(self):
        pass

    async def clear(self):
        keysdown = self._keysdown.copy()
        for key in keysdown:
            await self._send_one_key_event(key, down=False)
        return keysdown

    def sync_clear(self):
        keysdown = self._keysdown.copy()
        for key in keysdown:
            self._sync_send_one_key_event(key, down=False)
        return keysdown

    def close(self):
        pass
