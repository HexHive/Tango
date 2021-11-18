# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

from pkg_resources import parse_version
from models import kaitaistruct
from models.kaitaistruct import KaitaiStruct, KaitaiStream, ByteBufferKaitaiStream, BytesIO, SEEK_CUR, SEEK_END
from functools import partial
from .user_funcs import get_ip_address
from functools import cached_property
from copy import copy
from interaction import ReceiveInteraction
from interaction import TransmitInteraction


if parse_version(kaitaistruct.__version__) < parse_version('0.9'):
    raise Exception("Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s" % (kaitaistruct.__version__))

class Smtp(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None, _entropy=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._buffer = BytesIO()
        self._entropy = _entropy
        self.extended_commands = []
        self.has_greeted = False
        self.last_sent_command = u""
        self.client_address = get_ip_address()

    @cached_property
    def state_machine(self):
        return {
            u"start": {
                u"initial": partial(self._root.Handshake, None, _parent=self, _root=self, _entropy=self._entropy)
            }
        }

    def _save_checkpoint(self):
        self._save_extended_commands = copy(self.extended_commands)
        self._save_has_greeted = copy(self.has_greeted)
        self._save_last_sent_command = copy(self.last_sent_command)
        self._save_client_address = copy(self.client_address)

    def _restore_checkpoint(self):
        self.extended_commands = self._save_extended_commands
        self.has_greeted = self._save_has_greeted
        self.last_sent_command = self._save_last_sent_command
        self.client_address = self._save_client_address

    class Empty(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None, _entropy=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._buffer = BytesIO()
            self._entropy = _entropy

        def _read(self, *excludes):
            self._unnamed0 = self._io.read_bytes(0)

        def _write(self, io):
            io.write_bytes(io, self._unnamed0)

        def _generate(self, entropy, *excludes):
            if not (u"unnamed_0" in excludes):
                self._unnamed0 = self._io.generate_bytes(entropy, 0, included_term=None, pad=None, encoding=None)



    class Crlf(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None, _entropy=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._buffer = BytesIO()
            self._entropy = _entropy

        def _read(self, *excludes):
            self.newline = self._io.read_bytes(2)
            if not  ((u"newline" in excludes) or (self.newline == b"\x0D\x0A")) :
                raise kaitaistruct.ValidationNotEqualError(b"\x0D\x0A", self.newline, self._io, u"/types/crlf/seq/0")

        def _write(self, io):
            io.write_bytes(io, b"\x0D\x0A")

        def _generate(self, entropy, *excludes):
            if not (u"newline" in excludes):
                self.newline = b"\x0D\x0A"



    class ServerResponse(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None, _entropy=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._buffer = BytesIO()
            self._entropy = _entropy

        def _read(self, *excludes):
            self.lines = []
            i = 0
            while True:
                _t_lines = self._root.ServerResponse.ResponseLine(self._io, _parent=self, _root=self._root)
                _ = _t_lines
                self.lines.append(_)
                _t_lines._read()
                if _.begin.separator == u" ":
                    break
                i += 1
            self._unnamed1 = self._root.Empty(self._io, _parent=self, _root=self._root)
            self._unnamed1._read()
            self._root.last_sent_command = u""

        def _write(self, io):
            for i in range(len(self.lines)):
                self.lines[i]._write(io)

            self._unnamed1._write(io)

        def _generate(self, entropy, *excludes):
            if not (u"lines" in excludes):
                self.lines = []
                i = 0
                while True:
                    _ = self._root.ServerResponse.ResponseLine(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
                    self.lines.append(_)
                    self.lines[i]._generate(entropy)
                    if _.begin.separator == u" ":
                        break
                    i += 1

            if not (u"unnamed_1" in excludes):
                self._unnamed1 = self._root.Empty(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
                self.unnamed_1._generate(entropy)

            self._root.last_sent_command = u""

        class ResponseLine(KaitaiStruct):
            def __init__(self, _io, _parent=None, _root=None, _entropy=None):
                self._io = _io
                self._parent = _parent
                self._root = _root if _root else self
                self._buffer = BytesIO()
                self._entropy = _entropy

            def _read(self, *excludes):
                self.begin = self._root.ServerResponse.ResponseLine.Begin(self._io, _parent=self, _root=self._root)
                self.begin._read()
                _on = self._root.last_sent_command
                if _on == u"EHLO":
                    self.body = self._root.ServerResponse.ResponseLine.EhloOkRsp(self._io, _parent=self, _root=self._root)
                    self.body._read()
                else:
                    self.body = self._root.ServerResponse.ResponseLine.RawBody(self._io, _parent=self, _root=self._root)
                    self.body._read()

            def _write(self, io):
                self.begin._write(io)
                _on = self._root.last_sent_command
                if _on == u"EHLO":
                    self.body._write(io)
                else:
                    self.body._write(io)

            def _generate(self, entropy, *excludes):
                if not (u"begin" in excludes):
                    self.begin = self._root.ServerResponse.ResponseLine.Begin(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
                    self.begin._generate(entropy)

                if not (u"body" in excludes):
                    _on = self._root.last_sent_command
                    if _on == u"EHLO":
                        self.body = self._root.ServerResponse.ResponseLine.EhloOkRsp(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
                        self.body._generate(entropy)
                    else:
                        self.body = self._root.ServerResponse.ResponseLine.RawBody(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
                        self.body._generate(entropy)


            class Begin(KaitaiStruct):
                def __init__(self, _io, _parent=None, _root=None, _entropy=None):
                    self._io = _io
                    self._parent = _parent
                    self._root = _root if _root else self
                    self._buffer = BytesIO()
                    self._entropy = _entropy

                def _read(self, *excludes):
                    self.code = self._io.read_bytes(3)
                    self.separator = (self._io.read_bytes(1)).decode(u"ascii")
                    if not  ((u"separator" in excludes) or (self.separator in [u" ", u"-"])) :
                        raise kaitaistruct.ValidationSeqContainsError(self._io, u"/types/server_response/types/response_line/types/begin/seq/1")

                def _write(self, io):
                    io.write_bytes(io, self.code)
                    io.write_bytes(io, bytes(self.separator, encoding=u"ascii"))

                def _generate(self, entropy, *excludes):
                    if not (u"code" in excludes):
                        self.code = self._io.generate_bytes(entropy, 3, included_term=None, pad=None, encoding=None)

                    if not (u"separator" in excludes):
                        self.separator = entropy.choice([u" ", u"-"])



            class End(KaitaiStruct):
                def __init__(self, _io, _parent=None, _root=None, _entropy=None):
                    self._io = _io
                    self._parent = _parent
                    self._root = _root if _root else self
                    self._buffer = BytesIO()
                    self._entropy = _entropy

                def _read(self, *excludes):
                    self._unnamed0 = self._root.Crlf(self._io, _parent=self, _root=self._root)
                    self._unnamed0._read()

                def _write(self, io):
                    self._unnamed0._write(io)

                def _generate(self, entropy, *excludes):
                    if not (u"unnamed_0" in excludes):
                        self._unnamed0 = self._root.Crlf(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
                        self.unnamed_0._generate(entropy)



            class RawBody(KaitaiStruct):
                def __init__(self, _io, _parent=None, _root=None, _entropy=None):
                    self._io = _io
                    self._parent = _parent
                    self._root = _root if _root else self
                    self._buffer = BytesIO()
                    self._entropy = _entropy

                def _read(self, *excludes):
                    self.data = (self._io.read_bytes_term(13, False, False, True)).decode(u"ascii")
                    self._unnamed1 = self._root.Crlf(self._io, _parent=self, _root=self._root)
                    self._unnamed1._read()

                def _write(self, io):
                    io.write_bytes(io, bytes(self.data, encoding=u"ascii"))
                    self._unnamed1._write(io)

                def _generate(self, entropy, *excludes):
                    if not (u"data" in excludes):
                        self.data = (self._io.generate_bytes(entropy, entropy.randrange(1, 256), included_term=None, pad=None, encoding=u"ascii")).decode(u"ascii")

                    if not (u"unnamed_1" in excludes):
                        self._unnamed1 = self._root.Crlf(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
                        self.unnamed_1._generate(entropy)



            class EhloOkRsp(KaitaiStruct):
                def __init__(self, _io, _parent=None, _root=None, _entropy=None):
                    self._io = _io
                    self._parent = _parent
                    self._root = _root if _root else self
                    self._buffer = BytesIO()
                    self._entropy = _entropy

                def _read(self, *excludes):
                    _on = self._root.has_greeted
                    if _on == False:
                        self.msg = self._root.ServerResponse.ResponseLine.EhloOkRsp.EhloGreet(self._io, _parent=self, _root=self._root)
                        self.msg._read()
                    elif _on == True:
                        self.msg = self._root.ServerResponse.ResponseLine.EhloOkRsp.EhloLine(self._io, _parent=self, _root=self._root)
                        self.msg._read()
                    if not  ((u"msg" in excludes) or (self._parent.begin.code == b"\x32\x35\x30")) :
                        raise kaitaistruct.ValidationNotEqualError(b"\x32\x35\x30", self._parent.begin.code, self._io, u"/types/server_response/types/response_line/types/ehlo_ok_rsp/seq/0")

                def _write(self, io):
                    _on = self._root.has_greeted
                    if _on == False:
                        self.msg._write(io)
                    elif _on == True:
                        self.msg._write(io)

                def _generate(self, entropy, *excludes):
                    if not (u"msg" in excludes):
                        self._parent.begin.code = b"\x32\x35\x30"


                class Keyword(KaitaiStruct):
                    def __init__(self, _io, _parent=None, _root=None, _entropy=None):
                        self._io = _io
                        self._parent = _parent
                        self._root = _root if _root else self
                        self._buffer = BytesIO()
                        self._entropy = _entropy

                    def _read(self, *excludes):
                        self.c = []
                        i = 0
                        while True:
                            _ = self._io.read_bytes(1)
                            self.c.append(_)
                            if  ((self.c[-1][0] == 32) or (self.c[-1][0] == 13)) :
                                break
                            i += 1
                        if self.terminator == 13:
                            self.lf = self._io.read_bytes(1)
                            if not  ((u"lf" in excludes) or (self.lf == b"\x0A")) :
                                raise kaitaistruct.ValidationNotEqualError(b"\x0A", self.lf, self._io, u"/types/server_response/types/response_line/types/ehlo_ok_rsp/types/keyword/seq/1")

                        if self.terminator == 32:
                            self.params = (self._io.read_bytes_term(13, False, False, True)).decode(u"ascii")

                        if self.terminator == 32:
                            self.newline = self._root.Crlf(self._io, _parent=self, _root=self._root)
                            self.newline._read()


                    def _write(self, io):
                        for i in range(len(self.c)):
                            io.write_bytes(io, self.c[i])

                        if self.terminator == 13:
                            io.write_bytes(io, b"\x0A")

                        if self.terminator == 32:
                            io.write_bytes(io, bytes(self.params, encoding=u"ascii"))

                        if self.terminator == 32:
                            self.newline._write(io)


                    def _generate(self, entropy, *excludes):
                        if not (u"c" in excludes):
                            self.c = []
                            i = 0
                            while True:
                                _ = self._io.generate_bytes(entropy, 1, included_term=None, pad=None, encoding=None)
                                self.c.append(_)
                                if  ((self.c[-1][0] == 32) or (self.c[-1][0] == 13)) :
                                    break
                                i += 1

                        if  ((not (u"lf" in excludes)) and (self.terminator == 13)) :
                            self.lf = b"\x0A"

                        if  ((not (u"params" in excludes)) and (self.terminator == 32)) :
                            self.params = (self._io.generate_bytes(entropy, entropy.randrange(1, 256), included_term=None, pad=None, encoding=u"ascii")).decode(u"ascii")

                        if  ((not (u"newline" in excludes)) and (self.terminator == 32)) :
                            self.newline = self._root.Crlf(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
                            self.newline._generate(entropy)


                    @property
                    def name(self):
                        if hasattr(self, '_m_name'):
                            return self._m_name if hasattr(self, '_m_name') else None

                        self._m_name = (bytes(b for s in self.c for b in s)).decode(u"ascii")[0:-1]
                        return self._m_name if hasattr(self, '_m_name') else None

                    @property
                    def terminator(self):
                        if hasattr(self, '_m_terminator'):
                            return self._m_terminator if hasattr(self, '_m_terminator') else None

                        self._m_terminator = self.c[-1][0]
                        return self._m_terminator if hasattr(self, '_m_terminator') else None


                class EhloGreet(KaitaiStruct):
                    def __init__(self, _io, _parent=None, _root=None, _entropy=None):
                        self._io = _io
                        self._parent = _parent
                        self._root = _root if _root else self
                        self._buffer = BytesIO()
                        self._entropy = _entropy

                    def _read(self, *excludes):
                        self.domain = self._root.ServerResponse.ResponseLine.EhloOkRsp.Keyword(self._io, _parent=self, _root=self._root)
                        self.domain._read()
                        self._root.has_greeted = True

                    def _write(self, io):
                        self.domain._write(io)

                    def _generate(self, entropy, *excludes):
                        if not (u"domain" in excludes):
                            self.domain = self._root.ServerResponse.ResponseLine.EhloOkRsp.Keyword(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
                            self.domain._generate(entropy)

                        self._root.has_greeted = True


                class EhloLine(KaitaiStruct):
                    def __init__(self, _io, _parent=None, _root=None, _entropy=None):
                        self._io = _io
                        self._parent = _parent
                        self._root = _root if _root else self
                        self._buffer = BytesIO()
                        self._entropy = _entropy

                    def _read(self, *excludes):
                        self.keyword = self._root.ServerResponse.ResponseLine.EhloOkRsp.Keyword(self._io, _parent=self, _root=self._root)
                        self.keyword._read()
                        self._root.extended_commands = self._root.extended_commands + [self.keyword.name]

                    def _write(self, io):
                        self.keyword._write(io)

                    def _generate(self, entropy, *excludes):
                        if not (u"keyword" in excludes):
                            self.keyword = self._root.ServerResponse.ResponseLine.EhloOkRsp.Keyword(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
                            self.keyword._generate(entropy)

                        self._root.extended_commands = self._root.extended_commands + [self.keyword.name]





    class ClientCommand(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None, _entropy=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._buffer = BytesIO()
            self._entropy = _entropy

        def _read(self, *excludes):
            self.command = (self._io.read_bytes_term(13, False, False, True)).decode(u"ascii")
            if not  ((u"command" in excludes) or (self.command in [u"EHLO", u"MAIL"])) :
                raise kaitaistruct.ValidationSeqContainsError(self._io, u"/types/client_command/seq/0")
            self._root.last_sent_command = self.command
            _on = self.command
            if _on == u"EHLO":
                self.params = self._root.ClientCommand.EhloCmd(self._io, _parent=self, _root=self._root)
                self.params._read()
            elif _on == u"MAIL":
                self.params = self._root.ClientCommand.MailCmd(self._io, _parent=self, _root=self._root)
                self.params._read()
            else:
                self.params = self._root.Empty(self._io, _parent=self, _root=self._root)
                self.params._read()
            self.newline = self._io.read_bytes(2)
            if not  ((u"newline" in excludes) or (self.newline == b"\x0D\x0A")) :
                raise kaitaistruct.ValidationNotEqualError(b"\x0D\x0A", self.newline, self._io, u"/types/client_command/seq/2")

        def _write(self, io):
            io.write_bytes(io, bytes(self.command, encoding=u"ascii"))
            _on = self.command
            if _on == u"EHLO":
                self.params._write(io)
            elif _on == u"MAIL":
                self.params._write(io)
            else:
                self.params._write(io)
            io.write_bytes(io, b"\x0D\x0A")

        def _generate(self, entropy, *excludes):
            if not (u"command" in excludes):
                self.command = entropy.choice([u"EHLO", u"MAIL"])

            self._root.last_sent_command = self.command
            if not (u"params" in excludes):
                _on = self.command
                if _on == u"EHLO":
                    self.params = self._root.ClientCommand.EhloCmd(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
                    self.params._generate(entropy)
                elif _on == u"MAIL":
                    self.params = self._root.ClientCommand.MailCmd(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
                    self.params._generate(entropy)
                else:
                    self.params = self._root.Empty(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
                    self.params._generate(entropy)

            if not (u"newline" in excludes):
                self.newline = b"\x0D\x0A"


        class EhloCmd(KaitaiStruct):
            def __init__(self, _io, _parent=None, _root=None, _entropy=None):
                self._io = _io
                self._parent = _parent
                self._root = _root if _root else self
                self._buffer = BytesIO()
                self._entropy = _entropy

            def _read(self, *excludes):
                self.separator = self._io.read_bytes(1)
                if not  ((u"separator" in excludes) or (self.separator == b"\x20")) :
                    raise kaitaistruct.ValidationNotEqualError(b"\x20", self.separator, self._io, u"/types/client_command/types/ehlo_cmd/seq/0")
                self.addr = (self._io.read_bytes_term(13, False, False, True)).decode(u"ascii")
                if not  ((u"addr" in excludes) or (self.addr == self._root.client_address)) :
                    raise kaitaistruct.ValidationNotEqualError(self._root.client_address, self.addr, self._io, u"/types/client_command/types/ehlo_cmd/seq/1")

            def _write(self, io):
                io.write_bytes(io, b"\x20")
                io.write_bytes(io, bytes(self.addr, encoding=u"ascii"))

            def _generate(self, entropy, *excludes):
                if not (u"separator" in excludes):
                    self.separator = b"\x20"

                if not (u"addr" in excludes):
                    self.addr = self._root.client_address



        class MailCmd(KaitaiStruct):
            def __init__(self, _io, _parent=None, _root=None, _entropy=None):
                self._io = _io
                self._parent = _parent
                self._root = _root if _root else self
                self._buffer = BytesIO()
                self._entropy = _entropy

            def _read(self, *excludes):
                self.separator = self._io.read_bytes(1)
                if not  ((u"separator" in excludes) or (self.separator == b"\x20")) :
                    raise kaitaistruct.ValidationNotEqualError(b"\x20", self.separator, self._io, u"/types/client_command/types/mail_cmd/seq/0")
                self.frm = (self._io.read_bytes_term(58, False, True, True)).decode(u"ascii")
                if not  ((u"frm" in excludes) or (self.frm == u"FROM")) :
                    raise kaitaistruct.ValidationNotEqualError(u"FROM", self.frm, self._io, u"/types/client_command/types/mail_cmd/seq/1")

            def _write(self, io):
                io.write_bytes(io, b"\x20")
                io.write_bytes(io, bytes(self.frm, encoding=u"ascii"))
                io.write_u1(io, 58)

            def _generate(self, entropy, *excludes):
                if not (u"separator" in excludes):
                    self.separator = b"\x20"

                if not (u"frm" in excludes):
                    self.frm = u"FROM"




    class Handshake(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None, _entropy=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._buffer = BytesIO()
            self._entropy = _entropy

        def __inner_iter__(self, entropy):
            _rpos = self._buffer.tell()
            self._root._save_checkpoint()
            while True:
                _in = ReceiveInteraction()
                yield _in
                self._buffer.seek(0, SEEK_END)
                self._buffer.write(_in.data)
                self._buffer.seek(_rpos)
                try:
                    _io__raw_server_greet = KaitaiStream(self._buffer)
                    self.server_greet = self._root.ServerResponse(_io__raw_server_greet, _parent=self, _root=self._root)
                    self.server_greet._read()
                    break
                except EOFError:
                    self._root._restore_checkpoint()
            self.client_ehlo = self._root.ClientCommand(self._io, _parent=self, _root=self._root, _entropy=self._entropy)
            self.client_ehlo.command = u"EHLO"
            self.client_ehlo._generate(entropy, u"command")
            _io__raw_client_ehlo = KaitaiStream(BytesIO())
            self.client_ehlo._write(_io__raw_client_ehlo)
            _tmp__raw_client_ehlo = _io__raw_client_ehlo._io.getvalue()
            yield TransmitInteraction(data=_tmp__raw_client_ehlo)
            _rpos = self._buffer.tell()
            self._root._save_checkpoint()
            while True:
                _in = ReceiveInteraction()
                yield _in
                self._buffer.seek(0, SEEK_END)
                self._buffer.write(_in.data)
                self._buffer.seek(_rpos)
                try:
                    _io__raw_server_ehlo = KaitaiStream(self._buffer)
                    self.server_ehlo = self._root.ServerResponse(_io__raw_server_ehlo, _parent=self, _root=self._root)
                    self.server_ehlo._read()
                    break
                except EOFError:
                    self._root._restore_checkpoint()

        def __iter__(self):
            yield from self.__inner_iter__(self._entropy)



