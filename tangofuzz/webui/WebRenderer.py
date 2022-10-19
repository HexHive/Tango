from . import info, debug

from webui import WebDataLoader
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial
from threading import Thread
import os
import asyncio
import websockets

WWW_PATH = os.path.join(os.path.dirname(__file__), 'www')

class WebRenderer:
    def __init__(self, session, http_port=8080, ws_port=8081):
        handler = partial(SimpleHTTPRequestHandler, directory=WWW_PATH)
        while True:
            address = ('', http_port)
            try:
                self._httpd = HTTPServer(address, handler)
                break
            except OSError:
                http_port += 2
                ws_port += 2

        self._ws_port = ws_port
        self._session = session

        info(f"WebUI listening on http://localhost:{http_port}")

    def start(self):
        self._start_httpd()
        # self._start_watchdog()
        self._start_websockets()

    def _start_httpd(self):
        th = Thread(target=self._httpd.serve_forever)
        th.daemon = True
        th.start()

    def _start_watchdog(self):
        def shutdown_thread():
            # FIXME this is now an asyncio.Event which needs to be handled
            # differently, possibly through a callback?
            stopped.wait()
            self._httpd.shutdown()
        th = Thread(target=shutdown_thread)
        th.daemon = True
        th.start()

    def _start_websockets(self):
        async def websockets_worker():
            loop = asyncio.get_running_loop()
            loop.events = {}
            server = await websockets.serve(self._websocket_handler, '', self._ws_port)
            await server.wait_closed()

        def websockets_thread():
            asyncio.run(websockets_worker())

        th = Thread(target=websockets_thread)
        th.daemon = True
        th.start()

    async def _websocket_handler(self, websocket, path):
        debug("Web client initiated")
        loader = WebDataLoader(websocket, self._session)
        try:
            await asyncio.gather(*loader.tasks)
        except websockets.exceptions.ConnectionClosedOK:
            debug("Web client terminated")
