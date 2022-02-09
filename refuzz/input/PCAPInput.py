from input       import InputBase, PreparedInput
from scapy.all   import *
from typing      import Sequence, Tuple, BinaryIO, Union
from interaction import (InteractionBase,
                        TransmitInteraction,
                        ReceiveInteraction,
                        DelayInteraction)
import random
import time

class PCAPInput(PreparedInput):
    LAYER_SOURCE = {
        Ether: "src",
        IP: "src",
        TCP: "sport",
        UDP: "sport"
    }

    LAYER_DESTINATION = {
        Ether: "dst",
        IP: "dst",
        TCP: "dport",
        UDP: "dport"
    }

    DELAY_THRESHOLD = 1.0

    def __init__(self, pcap: Union[str, BinaryIO], interactions: Sequence[InteractionBase] = None,
            protocol: str = None):
        if interactions and not pcap:
            raise RuntimeError("'interactions' may not be specified without 'pcap'")

        read = False
        if pcap and not interactions:
            read = True

        self._pcap = pcap
        self._protocol = protocol

        if read:
            super().__init__(self.read_pcap())
        else:
            super().__init__(interactions)
            self.write_pcap()

    @classmethod
    def _try_identify_endpoints(cls, packet: Packet) -> Tuple:
        sender = []
        receiver = []
        for layer, src in cls.LAYER_SOURCE.items():
            if layer in packet:
                sender.append(getattr(packet.getlayer(layer), src))
        for layer, dst in cls.LAYER_DESTINATION.items():
            if layer in packet:
                receiver.append(getattr(packet.getlayer(layer), dst))
        if not sender or not receiver:
            raise RuntimeError("Could not identify endpoints in packet")
        return (tuple(sender), tuple(receiver))

    def read_pcap(self):
        if self._protocol in ("tcp", "udp"):
            layer = Raw
        else:
            layer = None
        plist = PcapReader(self._pcap).read_all()
        endpoints = []
        for p in plist:
            eps = self._try_identify_endpoints(p)
            for endpoint in eps:
                if endpoint not in endpoints:
                    endpoints.append(endpoint)
        if len(endpoints) != 2:
            raise RuntimeError(
                f"PCAP file has {len(endpoints)} endpoints (expected 2)"
            )

        if layer:
            plist = [p for p in plist if p.haslayer(layer)]

        tlast = plist[0].time
        for p in plist:
            # FIXME this operation is done previously, optimize
            src, dst = self._try_identify_endpoints(p)

            if layer:
                payload = bytes(p.getlayer(layer))
            else:
                payload = bytes(p)

            delay = p.time - tlast
            tlast = p.time
            if delay >= self.DELAY_THRESHOLD:
                yield DelayInteraction(float(delay))

            if src == endpoints[0]:
                interaction = TransmitInteraction(data=payload)
            else:
                interaction = ReceiveInteraction(data=payload)
            yield interaction

    def write_pcap(self):
        if self._protocol == "tcp":
            layer = TCP
            cli = random.randint(40000, 65534)
            srv = random.randint(cli + 1, 65535)
        elif self._protocol == "udp":
            layer = UDP
            cli = random.randint(40000, 65534)
            srv = random.randint(cli + 1, 65535)
        else:
            raise NotImplemented()

        cur_time = time.time()
        writer = PcapWriter(self._pcap)
        client_sent = False
        for interaction in self:
            if isinstance(interaction, DelayInteraction):
                if interaction._time >= self.DELAY_THRESHOLD:
                    cur_time += interaction._time
                continue
            elif isinstance(interaction, TransmitInteraction):
                src, dst = cli, srv
                client_sent = True
            elif isinstance(interaction, ReceiveInteraction):
                if not client_sent:
                    continue
                src, dst = srv, cli
            p = Ether(src='aa:aa:aa:aa:aa:aa', dst='aa:aa:aa:aa:aa:aa') / IP() / \
                    layer(**{self.LAYER_SOURCE[layer]: src, self.LAYER_DESTINATION[layer]: dst}) / \
                        Raw(load=interaction._data)
            p.time = cur_time
            writer.write(p)
