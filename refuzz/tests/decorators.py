from interaction import (ReceiveInteraction as rx,
                        TransmitInteraction as tx,
                        DelayInteraction as sleep)
from input import PreparedInput, SlicingDecorator, JoiningDecorator

def main():
    inp = PreparedInput()
    inp.append(tx(b'1234'))
    inp.append(rx(4, b'1234'))
    inp.append(sleep(1))

    sinp = SlicingDecorator(slice(1, None))(inp)
    asinp = JoiningDecorator(inp)(sinp)

    assert inp == inp
    assert sinp == sinp
    assert asinp == asinp
    assert inp != sinp
    assert inp != asinp
    assert sinp != asinp
    assert inp[1:] == sinp
    assert inp[1:] == asinp[:2]
    assert inp[1:] != asinp[:3]
    assert sinp == asinp[:2]
    assert sinp != asinp[:3]
    assert asinp[2:] != asinp[3:]
    assert asinp[3:] != asinp[2:]
    assert asinp == sinp + inp
    assert asinp != inp + sinp

if __name__ == '__main__':
    main()