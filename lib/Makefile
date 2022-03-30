CC=clang
CFLAGS := -fsanitize-coverage=$(COVERAGE),trace-pc-guard -fsanitize=address
CFLAGS += -fno-omit-frame-pointer -DUNW_LOCAL_ONLY -DSTACK_DEPTH=10
CFLAGS += -O3 -g -fPIC

all: libtangofuzz.o pytangofuzz.so

pytangofuzz.so: pytangofuzz.c
	$(CC) -O0 -g -fPIC -fPIE -shared $^ -o $@ -DSKIP_COUNTS

clean:
	rm -f *.o *.so
