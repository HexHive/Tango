from interaction import (ReceiveInteraction as rx,
                        TransmitInteraction as tx,
                        DelayInteraction as sleep)
from input import PreparedInput, CachingDecorator
from mutator import HavocMutator
from random import Random

def main():
    rand = Random()

    inp = PreparedInput()
    inp.append(tx(b'1234'))
    inp.append(rx(4, b'5678'))
    inp.append(sleep(1))

    inp_copy = inp[:]

    havoc1 = HavocMutator(rand)(inp)
    havoc1_cached = CachingDecorator()(havoc1, copy=True)
    havoc2 = HavocMutator(rand)(inp)
    havoc2_cached = CachingDecorator()(havoc2, copy=True)
    havoc3 = HavocMutator(rand)(havoc1)
    havoc3_cached = CachingDecorator()(havoc3, copy=True)
    havoc1_again = CachingDecorator()(havoc1, copy=True)

    assert havoc1 == havoc1
    assert havoc2 == havoc2
    assert havoc3 == havoc3
    assert havoc1_again == havoc1_again
    assert havoc1 == havoc1_cached
    assert havoc2 == havoc2_cached
    assert havoc3 == havoc3_cached
    assert havoc1 == havoc1_again
    assert havoc1_cached == havoc1_again

if __name__ == '__main__':
    main()