from webui import WebLogHandler
from profiler import ProfiledObjects
import networkx as nx
import json
import asyncio
import logging
import datetime
now = datetime.datetime.now

class WebDataLoader:
    NA_DATE = datetime.datetime.fromtimestamp(0)

    NODE_LINE_COLOR = (110, 123, 139)

    DEFAULT_NODE_PEN_WIDTH = 1.0
    NEW_NODE_PEN_WIDTH = 5

    DEFAULT_NODE_COLOR = (255, 255, 255)
    LAST_UPDATE_NODE_COLOR = (202, 225, 255)

    DEFAULT_EDGE_PEN_WIDTH = 1.0
    NEW_EDGE_PEN_WIDTH = 5

    DEFAULT_EDGE_COLOR = (0, 0, 0)
    LAST_UPDATE_EDGE_COLOR = (202, 225, 255)

    def __init__(self, websocket, last_update_fade_out=1.0):
        self._ws = websocket
        self._fade = last_update_fade_out

        ProfiledObjects['update_state'].listener(period=0.1)(self.update_graph)
        ProfiledObjects['execute_input'].listener(period=1)(self.update_stats)

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

    def update_graph(self, sm, state, ret):
        # update graph representation to be sent over WS
        # * color graph nodes and edges based on last_visit and added
        # * dump a DOT representation of the graph
        # * send over WS

        # first we get a copy so that we can re-assign node and edge attributes
        G = sm._graph.copy()

        for node, data in G.nodes(data=True):
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
            data['color'] = self.format_color(*self.NODE_LINE_COLOR)
            data['fillcolor'] = fillcolor
            data['penwidth'] = penwidth

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

            data.clear()
            data['color'] = color
            data['penwidth'] = penwidth

        A = nx.nx_agraph.to_agraph(G)
        A.node_attr['style'] = 'filled'
        dot = A.string()
        msg = json.dumps({
            'cmd': 'update_graph',
            'items': {
                'dot': dot
            }
        })

        asyncio.run(self._ws.send(msg))

    def update_stats(self, loader, input, sman, ret):
        # FIXME this is not thread-safe, and the websocket experience a data race
        stats = {'items': {}}
        for name, obj in ProfiledObjects.items():
            try:
                stats['items'][name] = obj.value
            except Exception:
                pass
        msg = json.dumps(stats | {'cmd': 'update_stats'})
        asyncio.run(self._ws.send(msg))