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

    @classmethod
    def _try_identify_endpoint(cls, packet: Packet) -> Tuple:
        endpoint = []
        for layer, src in cls.LAYER_SOURCE.items():
            if layer in packet:
                endpoint.append(getattr(packet.getlayer(layer), src))
        if not endpoint:
            raise RuntimeError("Could not identify endpoint in packet")
        return tuple(endpoint)

    def read_pcap(self):
        if self._protocol in ("tcp", "udp"):
            layer = Raw
        else:
            layer = None
        plist = PcapReader(self._pcap).read_all()
        endpoints = []
        for p in plist:
            endpoint = self._try_identify_endpoint(p)
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
            endpoint = self._try_identify_endpoint(p)

            if layer:
                payload = bytes(p.getlayer(layer))
            else:
                payload = bytes(p)

            delay = p.time - tlast
            tlast = p.time
            if delay >= self.DELAY_THRESHOLD:
                yield DelayInteraction(float(delay))

            if endpoint == endpoints[0]:
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
        for interaction in self:
            if isinstance(interaction, DelayInteraction):
                if interaction._time >= self.DELAY_THRESHOLD:
                    cur_time += interaction._time
                continue
            elif isinstance(interaction, TransmitInteraction):
                src, dst = cli, srv
            elif isinstance(interaction, ReceiveInteraction):
                src, dst = srv, cli
            p = Ether() / IP() / \
                    layer(**{self.LAYER_SOURCE[layer]: src, self.LAYER_DESTINATION[layer]: dst}) / \
                        Raw(load=interaction._data)
            p.time = cur_time
            writer.write(p)
