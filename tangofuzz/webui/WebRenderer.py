from . import info, debug

from webui import WebDataLoader
import os
import asyncio
import profiler
from aiohttp import web, web_urldispatcher
import logging

WWW_PATH = os.path.join(os.path.dirname(__file__), 'www')

class WebRenderer:
    def __init__(self, session, http_port=8080):
        self._http_port = http_port
        self._session = session

    async def run(self):
        await self._start_http_server(self._http_port)
        await profiler.ProfilingStoppedEvent.wait()
        await self._http_site.stop()

    async def _start_http_server(self, port):
        for name in ('access', 'server'):
            logging.getLogger(f'aiohttp.{name}').setLevel(logging.WARNING)

        server = web.Server(self.http_request_handler)
        runner = web.ServerRunner(server)
        await runner.setup()

        while True:
            try:
                self._http_site = web.TCPSite(runner, 'localhost', port)
                await self._http_site.start()
                break
            except OSError:
                port += 1
        info(f"WebUI listening on http://localhost:{port}")

    async def http_request_handler(self, request):
        if request.path == '/ws':
            return await self._handle_websocket(request)
        else:
            return await self._handle_static(request)

    async def _handle_static(self, request):
        resource = web_urldispatcher.StaticResource('', WWW_PATH)
        request.match_info, _ = await resource.resolve(request)
        if request.match_info is None:
            return web.HTTPNotFound()
        elif not request.match_info['filename']:
            request.match_info['filename'] = 'index.html'
        return await resource._handle(request)

    async def _handle_websocket(self, request):
        ws = web.WebSocketResponse(compress=False)
        await ws.prepare(request)
        data_loader = WebDataLoader(ws, self._session)
        gather_tasks = asyncio.gather(*data_loader.tasks)
        try:
            await gather_tasks
        except (RuntimeError, ConnectionResetError) as ex:
            debug(f'Websocket handler terminated ({ex=})')
        finally:
            gather_tasks.cancel()