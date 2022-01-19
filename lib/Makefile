CC=clang
CFLAGS=-fsanitize-coverage=$(COVERAGE),trace-pc-guard -O0 -g -fPIC
LDFLAGS=-lrt -O0 -g -fPIC -fPIE -pie
LDLIBS=-l:bind_wrapper.o -l:refuzz.o

all: bind_wrapper.o refuzz.o rebind.so

rebind.so: rebind.c
	$(CC) -O0 -g -fPIC -fPIE -shared $^ -o $@

clean:
	rm -f *.o *.so
