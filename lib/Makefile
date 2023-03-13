CC=clang-13
CFLAGS := -fsanitize-coverage=$(COVERAGE),trace-pc-guard
ifdef USE_ASAN
	CFLAGS += -fsanitize=address
endif
CFLAGS += -O3 -g -fPIC

.PHONY: all clean

all: libtangofuzz.o coverage.so

clean:
	rm -f *.o *.so

libtangofuzz.o: libtangofuzz.c clean

COV_CFLAGS := -O3 -fPIC -fPIE
ifdef SKIP_COUNTS
	COV_CFLAGS += -DSKIP_COUNTS
endif

coverage.so: coverage.c clean
	$(CC) ${COV_CFLAGS} -shared $< -o $@
