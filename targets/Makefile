export COVERAGE=func

export CC=clang
export CXX=clang++

CFLAGS := -fsanitize-coverage=$(COVERAGE),trace-pc-guard
CFLAGS += -fno-omit-frame-pointer
CFLAGS += -O3 -fPIC -fsanitize=address

export CFLAGS
export CXXFLAGS=$(CFLAGS)

LIBDIR=$(realpath ../lib)

export LDFLAGS=-Wl,-wrap,bind -Wl,-wrap,fork -fPIE -pie -L$(LIBDIR) -fsanitize=address
export LDLIBS=-l:libtangofuzz.o -lrt -lunwind

TARGETS=$(sort $(dir $(wildcard ./*/Makefile)))

$(info $$TARGETS is [${TARGETS}])

.PHONY: $(TARGETS) lib

all: $(TARGETS)

$(TARGETS): lib
	-$(MAKE) -C $@

lib:
	$(MAKE) -C $(LIBDIR) all
