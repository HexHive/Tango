from . import debug
from webui import WebLogHandler
from profiler import ProfiledObjects
from statemanager import StateMachine
import networkx as nx
import json
import asyncio
import asynctempfile
import logging
import datetime
import os
now = datetime.datetime.now

# A hacky fix for pydot to disable its crappy attribute getter/setter generator
import pydot
pydot.Common.create_attribute_methods = lambda *args, **kwargs: None

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

    def __init__(self, websocket, session, last_update_fade_out=1.0):
        self._ws = websocket
        self._session = session
        self._fade = last_update_fade_out

        self.tasks = []
        self.tasks.append(asyncio.create_task(
                ProfiledObjects['update_state'].listener(period=0.1)(self.update_graph)
            )
        )
        self.tasks.append(asyncio.create_task(
                ProfiledObjects['perform_interaction'].listener(period=1)(self.update_stats)
            )
        )

        self._sm = None

        # handler = WebLogHandler(websocket)
        # logging.addHandler(handler)

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

    async def update_graph(self, sm, *args, ret=None, **kwargs):
        # update graph representation to be sent over WS
        # * color graph nodes and edges based on last_visit and added
        # * dump a DOT representation of the graph
        # * send over WS

        if not self._sm:
            if isinstance(sm, StateMachine):
                self._sm = sm
            else:
                return

        # first we get a copy so that we can re-assign node and edge attributes
        try:
            # FIXME this is a hack: copying the graph might fail due to
            # concurrent access.
            G = self._sm._graph.copy()
        except Exception:
            return

        to_delete = []
        for node, data in G.nodes(data=True):
            if len(G.nodes) > 1 and len(G.in_edges(node)) == 0 \
                    and len(G.out_edges(node)) == 0:
                to_delete.append(node)
                continue
            age = (now() - data.get('last_visit', self.NA_DATE)).total_seconds()
            coeff = self.fade_coeff(self._fade, age)
            lerp = self.lerp_color(
                self.DEFAULT_NODE_COLOR,
                self.LAST_UPDATE_NODE_COLOR,
                coeff)
            fillcolor = self.format_color(*lerp)

            age = (now() - data.get('added', self.NA_DATE)).total_seconds()
            coeff = self.fade_coeff(self._fade, age)
            penwidth = self.lerp(
                self.DEFAULT_NODE_PEN_WIDTH,
                self.NEW_NODE_PEN_WIDTH,
                coeff)

            data.clear()
            data['fillcolor'] = fillcolor
            data['penwidth'] = penwidth
            if node == self._session._sman._strategy.target_state:
                data['color'] = self.format_color(*self.TARGET_LINE_COLOR)
                data['penwidth'] = self.NEW_NODE_PEN_WIDTH
            else:
                data['color'] = self.format_color(*self.NODE_LINE_COLOR)

        for node in to_delete:
            G.remove_node(node)

        for src, dst, data in G.edges(data=True):
            age = (now() - data.get('last_visit', self.NA_DATE)).total_seconds()
            coeff = self.fade_coeff(self._fade, age)
            lerp = self.lerp_color(
                self.DEFAULT_EDGE_COLOR,
                self.LAST_UPDATE_EDGE_COLOR,
                coeff)
            color = self.format_color(*lerp)

            age = (now() - data.get('added', self.NA_DATE)).total_seconds()
            coeff = self.fade_coeff(self._fade, age)
            penwidth = self.lerp(
                self.DEFAULT_EDGE_PEN_WIDTH,
                self.NEW_EDGE_PEN_WIDTH,
                coeff)
            label = f"min={len(data['minimized'])}"

            data.clear()
            data['color'] = color
            data['penwidth'] = penwidth
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
        # FIXME this is not thread-safe, and the websocket experience a data race
        stats = {'items': {}}
        for name, obj in ProfiledObjects.items():
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