CC := clang-13
CXX := clang++-13
AR := llvm-ar-13

CFLAGS := -O3 -fPIC -I.
PYBINDING_CFLAGS := $(CFLAGS) -fPIE -shared
ifdef SKIP_COUNTS
	PYBINDING_CFLAGS += -DSKIP_COUNTS
endif

TRACEPCGUARD_FLAG := trace-pc-guard
TRACECMP_FLAG := trace-cmp

SANCOV_FLAGS := $(TRACEPCGUARD_FLAG)
ifdef USE_CMPLOG
	SANCOV_FLAGS += $(TRACECMP_FLAG)
	CFLAGS += -DUSE_CMPLOG
endif
space:=$(subst ,, )
comma:=,
SANCOV_FLAGS := $(subst $(space),$(comma),$(SANCOV_FLAGS))

CFLAGS += -fsanitize-coverage=$(SANCOV_FLAGS)
CFLAGS += -fsanitize-coverage-ignorelist=sancov_ignorelist.txt
CXXFLAGS := $(CFLAGS)

.PHONY: all clean

all: libtangofuzz.a pyfeaturemap.so

clean:
	rm -f *.o *.a *.so

libtangofuzz.a: tracer.o frksrv.o wrappers.o
	$(AR) -rcs $@ $^

pyfeaturemap.so: pyfeaturemap.c
	$(CC) ${PYBINDING_CFLAGS} $< -o $@
