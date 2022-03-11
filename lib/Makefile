CC=clang
CFLAGS=-fsanitize-coverage=$(COVERAGE),trace-pc-guard -O3 -fPIC

all: libtangofuzz.o pytangofuzz.so

pytangofuzz.so: pytangofuzz.c
	$(CC) -O0 -g -fPIC -fPIE -shared $^ -o $@

clean:
	rm -f *.o *.so
