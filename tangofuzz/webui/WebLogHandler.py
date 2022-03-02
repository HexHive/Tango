from logging import Handler
import logging
import asyncio
import json

class WebLogHandler(Handler):
    def __init__(self, websocket):
        self._ws = websocket
        self.setLevel(logging.INFO)

    def emit(self, record):
        msg = json.dumps({'cmd': 'update_log', 'msg': record})
        asyncio.run(self._ws.send(msg))