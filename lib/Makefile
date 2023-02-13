CC=clang
CFLAGS := -fsanitize-coverage=$(COVERAGE),trace-pc-guard
ifdef USE_ASAN
	CFLAGS += -fsanitize=address
endif
CFLAGS += -O3 -g -fPIC

.PHONY: all clean

all: libtangofuzz.o pytangofuzz.so

clean:
	rm -f *.o *.so

libtangofuzz.o: libtangofuzz.c clean

# -march=skylake-avx512 -mprefer-vector-width=512
pytangofuzz.so: pytangofuzz.c clean
	$(CC) -O3 \
		-fPIC -fPIE -shared $< -o $@
