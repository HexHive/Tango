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

pytangofuzz.so: pytangofuzz.c clean
	$(CC) -O0 -g -fPIC -fPIE -shared $< -o $@ -DSKIP_COUNTS
