export CC=clang
export CXX=clang++
TANGO_LIBDIR ?= $(realpath ../lib/)

# one of: edge, bb, func
COVERAGE ?= edge
TRACEPCGUARD_FLAG := $(COVERAGE),trace-pc-guard
TRACECMP_FLAG := trace-cmp

CFLAGS := -fPIC
LDFLAGS := -fPIE -pie -L$(TANGO_LIBDIR)/tango
SANCOV_FLAGS := $(TRACEPCGUARD_FLAG)
ifdef USE_CMPLOG
	SANCOV_FLAGS += $(TRACECMP_FLAG)
	CFLAGS += -DUSE_CMPLOG
endif
space:=$(subst ,, )
comma:=,
SANCOV_FLAGS := $(subst $(space),$(comma),$(SANCOV_FLAGS))

ifndef DISABLE_COV
	CFLAGS += -fsanitize-coverage=$(SANCOV_FLAGS)
endif

ifndef DISABLE_TANGOLIB
	LDFLAGS += -Wl,-wrap,bind -Wl,-wrap,fork -Wl,-wrap,random
	# LDLIBS=-Wl,-whole-archive -l:libtangofuzz.a -Wl,-no-whole-archive -lrt -lc++
	LDLIBS= \
		$(TANGO_LIBDIR)/tango/tracer.o \
		$(TANGO_LIBDIR)/tango/frksrv.o \
		$(TANGO_LIBDIR)/tango/wrappers.o \
		-lrt -lc++
endif

ifdef USE_ASAN
	CFLAGS += -fsanitize=address
	LDFLAGS += -fsanitize=address
endif

ifdef DEBUG
	CFLAGS += -O0 -g -fno-inline
else
	CFLAGS += -O3
endif

export CFLAGS
export CXXFLAGS=$(CFLAGS)
export LDFLAGS
export LDLIBS

###

TARGETS=$(sort $(dir $(wildcard ./*/Makefile)))

$(info $$TARGETS is [${TARGETS}])

.PHONY: $(TARGETS) lib

all: $(TARGETS)

$(TARGETS): lib
	-$(MAKE) -C $@

lib:
	$(MAKE) -C $(TANGO_LIBDIR)/tango all
