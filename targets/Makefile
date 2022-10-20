# one of: edge, bb, func
export COVERAGE=bb

export CC=clang
export CXX=clang++

CFLAGS := -fsanitize-coverage=$(COVERAGE),trace-pc-guard
CFLAGS += -O3 -fPIC
ifdef USE_ASAN
	CFLAGS += -fsanitize=address
endif

export CFLAGS
export CXXFLAGS=$(CFLAGS)

LIBDIR=$(realpath ../lib)

LDFLAGS := -fPIE -pie -L$(LIBDIR)
ifdef USE_ASAN
	LDFLAGS += -fsanitize=address
endif
LDFLAGS += -Wl,-wrap,bind -Wl,-wrap,fork -Wl,-wrap,random

export LDFLAGS
export LDLIBS=-l:libtangofuzz.o -lrt

TARGETS=$(sort $(dir $(wildcard ./*/Makefile)))

$(info $$TARGETS is [${TARGETS}])

.PHONY: $(TARGETS) lib

all: $(TARGETS)

$(TARGETS): lib
	-$(MAKE) -C $@

lib:
	$(MAKE) -C $(LIBDIR) all
