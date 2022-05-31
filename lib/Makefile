CC=clang
CFLAGS := -fsanitize-coverage=$(COVERAGE),trace-pc-guard -fsanitize=address
CFLAGS += -O3 -g -fPIC

all: libtangofuzz.o pytangofuzz.so

pytangofuzz.so: pytangofuzz.c
	$(CC) -O0 -g -fPIC -fPIE -shared $^ -o $@ -DSKIP_COUNTS

clean:
	rm -f *.o *.so
