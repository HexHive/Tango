export COVERAGE=edge
export CC=clang
export CXX=clang++
export CFLAGS=-fsanitize-coverage=$(COVERAGE),trace-pc-guard -O0 -g -fPIC
export CXXFLAGS=$(CFLAGS)

LIBDIR=$(realpath ../lib)

export LDFLAGS=-Wl,-wrap,bind -Wl,-wrap,fork -fPIE -pie -L$(LIBDIR)
export LDLIBS=-l:libtangofuzz.o -lrt

TARGETS=$(sort $(dir $(wildcard ./*/Makefile)))

$(info $$TARGETS is [${TARGETS}])

.PHONY: $(TARGETS) lib

all: $(TARGETS)

$(TARGETS): lib
	-$(MAKE) -C $@

lib:
	$(MAKE) -C $(LIBDIR) all