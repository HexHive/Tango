from models import smtp
from random import Random

def main():
    entropy = Random()
    p = smtp.Smtp(None, _entropy=entropy)

    assert(test_handshake(p))

def test_handshake(p):
    try:
        hs = p.state_machine['start']['initial']()
        ihs = iter(hs)

        # receive  server greeting
        _ = next(ihs)
        _.data = b"220 mx.google.com ESMTP q8si1038396vcq.58 - gsmtp\r\n"

        # transmit EHLO
        _ = next(ihs)

        # receive partial EHLO OK
        _ = next(ihs)
        _.data = b"250-mx.google.com at your service, [108.39.81.51]\r\n" \
                 b"250-SIZE 35882577\r\n" \
                 b"250-8BITMIME\r\n" \
                 b"250-STARTTLS\r\n"

        _ = next(ihs)
        _.data = b"250-ENHANCEDSTATUSCODES\r\n" \
                 b"250-PIPELINING\r\n" \
                 b"250-CHUNKING\r\n" \
                 b"250 SMTPUTF8\r\n"

        # StopIteration
        _ = next(ihs)
    except StopIteration:
        assert all(map(lambda x: x in p.extended_commands,
                (
                    'SIZE',
                    '8BITMIME',
                    'STARTTLS',
                    'ENHANCEDSTATUSCODES',
                    'PIPELINING',
                    'CHUNKING',
                    'SMTPUTF8',
                )
            )
        )
        assert p.has_greeted
        return True
    return False

if __name__ == '__main__':
    main()