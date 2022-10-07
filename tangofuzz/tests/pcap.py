from interaction import (ReceiveInteraction as rx,
                        TransmitInteraction as tx,
                        DelayInteraction as sleep)
from input import PCAPInput
from dataio import TCPChannelFactory
from io import BytesIO, SEEK_SET

def main():
    interactions = []
    interactions.append(tx(b'1234'))
    interactions.append(rx(4, b'5678'))
    interactions.append(sleep(2))
    interactions.append(tx(b'9101112'))

    io = BytesIO()
    ch_env = TCPChannelFactory(1, None, 0)
    pcap = PCAPInput(io, interactions=interactions, protocol="tcp")

    for i in range(10):
        io.seek(0, SEEK_SET)
        for left, right in zip(pcap.read_pcap(), interactions):
            assert left == right