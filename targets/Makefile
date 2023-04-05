export CC=clang-13
export CXX=clang++-13
TANGO_LIBDIR ?= $(realpath ../lib/)

# one of: edge, bb, func
COVERAGE ?= edge
TRACEPCGUARD_FLAG := $(COVERAGE),trace-pc-guard
TRACECMP_FLAG := trace-cmp

SANCOV_FLAGS := $(TRACEPCGUARD_FLAG)
ifdef USE_CMPLOG
	SANCOV_FLAGS += $(TRACECMP_FLAG)
endif
space:=$(subst ,, )
comma:=,
SANCOV_FLAGS := $(subst $(space),$(comma),$(SANCOV_FLAGS))

CFLAGS := -O3 -fPIC
CFLAGS += -fsanitize-coverage=$(SANCOV_FLAGS)
ifdef USE_ASAN
	CFLAGS += -fsanitize=address
endif

export CFLAGS
export CXXFLAGS=$(CFLAGS)

LDFLAGS := -fPIE -pie -L$(TANGO_LIBDIR)/tango
ifdef USE_ASAN
	LDFLAGS += -fsanitize=address
endif
LDFLAGS += -Wl,-wrap,bind -Wl,-wrap,fork -Wl,-wrap,random

export LDFLAGS
# export LDLIBS=-Wl,-whole-archive -l:libtangofuzz.a -Wl,-no-whole-archive -lrt -lc++
export LDLIBS= \
	$(TANGO_LIBDIR)/tango/tracer.o \
	$(TANGO_LIBDIR)/tango/frksrv.o \
	$(TANGO_LIBDIR)/tango/wrappers.o \
	-lrt -lc++

###

TARGETS=$(sort $(dir $(wildcard ./*/Makefile)))

$(info $$TARGETS is [${TARGETS}])

.PHONY: $(TARGETS) lib

all: $(TARGETS)

$(TARGETS): lib
	-$(MAKE) -C $@

lib:
	$(MAKE) -C $(TANGO_LIBDIR)/tango all
