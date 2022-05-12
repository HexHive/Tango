from __future__ import annotations
from networkio import ChannelBase, ChannelFactoryBase
from common import ChannelSetupException

from dataclasses import dataclass
from Xlib import XK, display, ext, X, protocol
import sys
import time

@dataclass
class X11ChannelFactory(ChannelFactoryBase):
    protocol: str = 'x11'

    def create(self, program_name=None, window_name=None, *args, **kwargs) -> X11Channel:
        return X11Channel(program_name=program_name, window_name=window_name, \
                            timescale=self.timescale)

class X11Channel(ChannelBase):
    WINDOW_POLL_RETRIES = 10
    WINDOW_POLL_WAIT = 1.0

    def __init__(self, *, program_name, window_name, **kwargs):
        super().__init__(**kwargs)
        self._display = display.Display()

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

    def send(self, key, release=False):
        keysym = XK.string_to_keysym(key)
        keycode = self._display.keysym_to_keycode(keysym)

        event_ctor = protocol.event.KeyRelease if release else protocol.event.KeyPress

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

    def receive(self):
        pass

    def close(self):
        pass