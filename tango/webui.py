from . import info, debug

from tango.core import (FuzzerSession, get_profiler, get_all_profilers)
from tango.common import AsyncComponent, get_session_task_group

from collections import defaultdict
from aiohttp import web, web_urldispatcher
from functools import partial
import os
import asyncio
import logging
import networkx as nx
import json
import asyncio
import asynctempfile
import logging
import datetime
import errno
import os

# A hacky fix for pydot to disable its crappy attribute getter/setter generator
import pydot
pydot.Common.create_attribute_methods = lambda *args, **kwargs: None
now = datetime.datetime.now

__all__ = ['WebRenderer', 'WebDataLoader', 'create_svg']

WWW_PATH = os.path.join(os.path.dirname(__file__), 'www')

class WebRenderer(AsyncComponent, component_type='webui',
        capture_paths=['webui.*'], capture_components={'session'},
        catch_all=True):
    def __init__(self, session: FuzzerSession, *,
            http_host: str='localhost', http_port: int=8080,
            www_path: str=WWW_PATH,
            draw_graph: bool=True,
            bubble_update_period: float=0.1, bubble_reload_period: float=1,
            **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._http_host = http_host
        self._http_port = http_port
        self._www_path = www_path
        self._hitcounter = defaultdict(int)

        self._draw_graph = draw_graph
        self._bubble_update_period = bubble_update_period
        self._bubble_reload_period = bubble_reload_period

        self._webui_kwargs = kwargs | {'draw_graph': draw_graph}

    def get_webui_factory(self):
        return partial(WebDataLoader, **self._webui_kwargs)

    def setup_counters(self):
        async def update(obj, *args, ret, **kwargs):
            if isinstance(state := ret, tuple):
                # StateGraph.update_state returns a tuple
                state, _ = ret
            self._hitcounter[state] += 1
            min_hits = min(self._hitcounter.values())
            self._hitcounter.update({
                    state: state_hits - min_hits + 1
                    for state, state_hits in self._hitcounter.items()
                })

        tg = get_session_task_group()
        if self._draw_graph and all(x >= 0 for x in
                (self._bubble_update_period, self._bubble_reload_period)):
            tg.create_task(
                get_profiler('update_state').listener(
                    period=self._bubble_update_period)(update))
            tg.create_task(
                get_profiler('reload_target').listener(
                    period=self._bubble_reload_period)(update))

    async def run(self):
        self.setup_counters()
        await self._start_http_server(self._http_host, self._http_port)
        try:
            # dummy future object to await indefinitely
            await asyncio.Future()
        except asyncio.CancelledError:
            await self._http_site.stop()

    async def _start_http_server(self, host, port):
        for name in ('access', 'server'):
            logging.getLogger(f'aiohttp.{name}').setLevel(logging.WARNING)

        server = web.Server(self.http_request_handler)
        runner = web.ServerRunner(server)
        await runner.setup()

        while True:
            try:
                self._http_site = web.TCPSite(runner, host, port)
                await self._http_site.start()
                break
            except OSError:
                port += 1
        info(f"WebUI listening on http://{host}:{port}")

    async def http_request_handler(self, request):
        if request.path == '/ws':
            return await self._handle_websocket(request)
        else:
            return await self._handle_static(request)

    async def _handle_static(self, request):
        resource = web_urldispatcher.StaticResource('', self._www_path)
        request.match_info, _ = await resource.resolve(request)
        if request.match_info is None:
            return web.HTTPNotFound()
        elif not request.match_info['filename']:
            request.match_info['filename'] = 'index.html'
        return await resource._handle(request)

    async def _handle_websocket(self, request):
        async def _handler():
            ws = web.WebSocketResponse(compress=False)
            await ws.prepare(request)

            data_loader = self.get_webui_factory()(
                ws, self._session, self._hitcounter)
            gather_tasks = asyncio.gather(*data_loader.tasks)
            try:
                await gather_tasks
            except (RuntimeError, ConnectionResetError) as ex:
                debug(f'Websocket handler terminated ({ex=})')
            finally:
                gather_tasks.cancel()
        await get_session_task_group().create_task(_handler())

class WebDataLoader:
    NA_DATE = datetime.datetime.fromtimestamp(0)

    NODE_LINE_COLOR = (110, 123, 139)
    TARGET_LINE_COLOR = (255, 0, 0)

    DEFAULT_NODE_PEN_WIDTH = 1.0
    NEW_NODE_PEN_WIDTH = 5

    DEFAULT_NODE_COLOR = (255, 255, 255)
    LAST_UPDATE_NODE_COLOR = (202, 225, 255)

    DEFAULT_EDGE_PEN_WIDTH = 1.0
    NEW_EDGE_PEN_WIDTH = 5

    DEFAULT_EDGE_COLOR = (0, 0, 0)
    LAST_UPDATE_EDGE_COLOR = (202, 225, 255)

    def __init__(self, websocket, session, hitcounter, *,
            draw_graph: bool,
            draw_update_period: float=0.1, draw_reload_period: float=1,
            draw_update_fadeout: float=1, stats_update_period: float=1):
        self._ws = websocket
        self._session = session
        self._hitcounter = hitcounter

        self._draw_update_fadeout = draw_update_fadeout

        self._node_added = {}
        self._node_visited = {}
        self._edge_added = {}
        self._edge_visited = {}

        self.tasks = []

        if draw_graph and all(x >= 0 for x in
                (draw_update_period, draw_reload_period, draw_update_fadeout)):
            self.tasks.append(asyncio.create_task(
                    get_profiler('update_state').listener(
                        period=draw_update_period)(self.track_node)))
            self.tasks.append(asyncio.create_task(
                    get_profiler('update_transition').listener(
                        period=draw_update_period)(self.track_edge)))
            self.tasks.append(asyncio.create_task(
                    get_profiler('reload_target').listener(
                        period=draw_reload_period)(self.update_graph)))

        if stats_update_period >= 0:
            self.tasks.append(asyncio.create_task(
                    get_profiler('perform_instruction').listener(
                        period=stats_update_period)(self.update_stats)))

    @classmethod
    def format_color(cls, r, g, b, a=None):
        if a is not None:
            return '#{:02X}{:02X}{:02X}{:02X}'.format(r, g, b, a)
        else:
            return '#{:02X}{:02X}{:02X}'.format(r, g, b)

    @classmethod
    def lerp(cls, value1, value2, coeff):
        return value1 + (value2 - value1) * coeff

    @classmethod
    def lerp_color(cls, color1, color2, coeff):
        return tuple(map(lambda c: int(cls.lerp(c[0], c[1], coeff)), zip(color1, color2)))

    @classmethod
    def fade_coeff(cls, fade, value):
        return max(0, (fade - value) / fade)

    @property
    def fresh_graph(self):
        H = self._session._explorer.tracker.state_graph
        G = H.copy(fresh=True)
        # we also return a reference to the original in case attribute access is
        # needed
        return G, H

    async def track_node(self, *args, ret, **kwargs):
        state, new = ret
        now = datetime.datetime.now()
        if new:
            self._node_added[state] = now
        self._node_visited[state] = now
        await self.update_graph()

    async def track_edge(self, *args, ret, **kwargs):
        src, dst, new = ret
        now = datetime.datetime.now()
        if new:
            self._edge_added[(src, dst)] = now
        self._edge_visited[(src, dst)] = now
        await self.update_graph()

    async def update_graph(self, *args, **kwargs):
        # update graph representation to be sent over WS
        # * color graph nodes and edges based on last_visit and added
        # * dump a DOT representation of the graph
        # * send over WS

        # first we get a copy so that we can re-assign node and edge attributes
        G, H = self.fresh_graph

        to_delete = []
        for node, data in G.nodes(data=True):
            if len(G.nodes) > 1 and len(G.in_edges(node)) == 0 \
                    and len(G.out_edges(node)) == 0:
                to_delete.append(node)
                continue
            age = (now() - self._node_visited.get(node, self.NA_DATE)).total_seconds()
            coeff = self.fade_coeff(self._draw_update_fadeout, age)
            lerp = self.lerp_color(
                self.DEFAULT_NODE_COLOR,
                self.LAST_UPDATE_NODE_COLOR,
                coeff)
            fillcolor = self.format_color(*lerp)

            age = (now() - self._node_added.get(node, self.NA_DATE)).total_seconds()
            coeff = self.fade_coeff(self._draw_update_fadeout, age)
            penwidth = self.lerp(
                self.DEFAULT_NODE_PEN_WIDTH,
                self.NEW_NODE_PEN_WIDTH,
                coeff)

            data.clear()
            data['fillcolor'] = fillcolor
            data['penwidth'] = penwidth
            if node == self._session._strategy.target_state:
                data['color'] = self.format_color(*self.TARGET_LINE_COLOR)
                data['penwidth'] = self.NEW_NODE_PEN_WIDTH
            else:
                data['color'] = self.format_color(*self.NODE_LINE_COLOR)
            data['width'] = 0.75 * self._hitcounter[node]
            data['height'] = 0.5 * self._hitcounter[node]
            data['fontsize'] = 14 * self._hitcounter[node]
            data['penwidth'] *= self._hitcounter[node]

        for node in to_delete:
            G.remove_node(node)

        for src, dst, data in G.edges(data=True):
            age = (now() - self._edge_visited.get((src, dst), self.NA_DATE)).total_seconds()
            coeff = self.fade_coeff(self._draw_update_fadeout, age)
            lerp = self.lerp_color(
                self.DEFAULT_EDGE_COLOR,
                self.LAST_UPDATE_EDGE_COLOR,
                coeff)
            color = self.format_color(*lerp)

            age = (now() - self._edge_added.get((src, dst), self.NA_DATE)).total_seconds()
            coeff = self.fade_coeff(self._draw_update_fadeout, age)
            penwidth = self.lerp(
                self.DEFAULT_EDGE_PEN_WIDTH,
                self.NEW_EDGE_PEN_WIDTH,
                coeff)

            state = dst
            data.clear()
            data['color'] = color
            data['penwidth'] = penwidth * self._hitcounter[state]
            if 'minimized' in H.edges[src, dst]:
                label = f"min={len(H.edges[src, dst]['minimized'].flatten())}"
                data['label'] = label

        G.graph["graph"] = {'rankdir': 'LR'}
        G.graph["node"] = {'style': 'filled'}
        P = nx.nx_pydot.to_pydot(G)
        svg = await create_svg(P)

        msg = json.dumps({
            'cmd': 'update_graph',
            'items': {
                # 'dot': dot,
                'svg': svg
            }
        })

        await self._ws.send_str(msg)

    async def update_stats(self, *args, ret=None, **kwargs):
        stats = {'items': {}}
        for name, obj in get_all_profilers():
            try:
                stats['items'][name] = obj.value
            except Exception:
                pass
        msg = json.dumps(stats | {'cmd': 'update_stats'})
        await self._ws.send_str(msg)

async def call_graphviz(program, arguments, **kwargs):
    if arguments is None:
        arguments = []

    env = {
        "PATH": os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
    }

    process = await asyncio.create_subprocess_exec(program, *arguments,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs
    )

    stdout_data, stderr_data = await process.communicate()
    return stdout_data, stderr_data, process

async def create_svg(P, prog=None, encoding='utf-8'):
    # temp file
    async with asynctempfile.NamedTemporaryFile('wt', encoding=encoding) as f:
        await f.write(P.to_string())
        await f.flush()

        prog = 'dot'
        arguments = ('-Tsvg', f.name)
        try:
            stdout_data, stderr_data, process = await call_graphviz(
                program=prog,
                arguments=arguments
            )
        except OSError as e:
            if e.errno == errno.ENOENT:
                args = list(e.args)
                args[1] = '"{prog}" not found in path.'.format(prog=prog)
                raise OSError(*args)
            else:
                raise

    if process.returncode != 0:
        # FIXME because ptrace.debugger reaps all children with waitpid(-1), the
        # asyncio loop child watcher does not get the chance to read off the
        # exit status of its children, and returns 255 by default.
        message = (
            '"{prog}" with args {arguments} returned code: {code}\n\n'
            "stdout, stderr:\n {out}\n{err}\n"
        ).format(
            prog=prog,
            arguments=arguments,
            code=process.returncode,
            out=stdout_data,
            err=stderr_data,
        )
        debug(message)

    return stdout_data.decode(encoding=encoding)
