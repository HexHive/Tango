export CC  = clang-13
export CXX = clang++-13

.PHONY: all clean tango libfaketime

all: tango libfaketime

clean:
	$(MAKE) -C tango clean
	$(MAKE) -C libfaketime distclean

libfaketime:
	CFLAGS=-DFAIL_PRE_INIT_CALLS $(MAKE) -C libfaketime all

tango:
	$(MAKE) -C tango all
