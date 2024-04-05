export CC  = clang
export CXX = clang++

.PHONY: all clean tango libfaketime

all: tango libfaketime

clean:
	$(MAKE) -C tango clean
	$(MAKE) -C libfaketime distclean

libfaketime:
	FAKETIME_COMPILE_CFLAGS="-DFAIL_PRE_INIT_CALLS -Wno-error" $(MAKE) -C libfaketime all

tango:
	$(MAKE) -C tango all
