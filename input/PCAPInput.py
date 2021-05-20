from input       import PreparedInput
from loader      import ChannelFactoryBase, TCPChannelFactory, UDPChannelFactory
from scapy.all   import *
from typing      import Sequence, Tuple
from interaction import (InteractionBase,
                        TransmitInteraction,
                        ReceiveInteraction,
                        DelayInteraction)
from collections import OrderedSet

class PCAPInput(PreparedInput):
    def __init__(self, pcap: str, ch_env: ChannelFactoryBase):
        super().__init__(self)
        if isinstance(ch_env, (TCPChannelFactory, UDPChannelFactory)):
            layer = Raw
        else:
            layer = None
        self._extract_layer(pcap, layer)

    @classmethod
    def _try_identify_endpoint(cls, packet: Packet) -> Tuple:
        endpoint = []
        layers = {
            Ether: "src",
            IP: "src",
            TCP: "sport",
            UDP: "sport"
        }
        for layer, src in :
            if layer in p:
                endpoint.append(getattr(p.getlayer(layer), src))
        if not endpoint:
            raise RuntimeError("Could not identify endpoint in packet")
        return tuple(endpoint)

    def _extract_layer(self, pcap: str, layer = None: Packet)
        plist = rdpcap(pcap)
        endpoints = OrderedSet(self._try_identify_endpoint(p) for p in plist)
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

            if endpoint == endpoints[0]:
                interaction = TransmitInteraction(data=payload)
            else:
                interaction = ReceiveInteraction(data=payload)
            self._interactions.append(interaction)

            delay = p.time - tlast
            tlast = p.time
            if delay >= 1E-3:
                self._interactions.append(DelayInteraction(float(delay)))