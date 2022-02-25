from webui import WebDataLoader
from profiler import ProfilingStoppedEvent as stopped
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial
from threading import Thread
import os
import asyncio
import websockets

WWW_PATH = os.path.join(os.path.dirname(__file__), 'www')

class WebRenderer:
    def __init__(self, session, http_port=8080, ws_port=8081):
        address = ('', http_port)
        handler = partial(SimpleHTTPRequestHandler, directory=WWW_PATH)
        self._httpd = HTTPServer(address, handler)
        self._ws_port = ws_port
        self._session = session

    def start(self):
        self._start_httpd()
        self._start_watchdog()
        self._start_websockets()

    def _start_httpd(self):
        th = Thread(target=self._httpd.serve_forever)
        th.daemon = True
        th.start()

    def _start_watchdog(self):
        def shutdown_thread():
            stopped.wait()
            self._httpd.shutdown()
        th = Thread(target=shutdown_thread)
        th.daemon = True
        th.start()

    def _start_websockets(self):
        async def websockets_worker():
            async with websockets.serve(self._websocket_handler, '', self._ws_port):
                await asyncio.Future()
        def websockets_thread():
            asyncio.run(websockets_worker())
        th = Thread(target=websockets_thread)
        th.daemon = True
        th.start()

    async def _websocket_handler(self, websocket, path):
        WebDataLoader(websocket, self._session)
        await asyncio.Future()
