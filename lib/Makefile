CC=clang
CFLAGS=-fsanitize-coverage=edge,trace-pc-guard -O0 -g -fPIC
LDFLAGS=-lrt -O0 -g -Wl,-wrap,bind -Wl,-wrap,fork -fPIC -fPIE -pie

all: bind_wrapper.o refuzz.o rebind.so demo_server

demo_server: demo_server.o bind_wrapper.o refuzz.o

rebind.so: rebind.c
	$(CC) -O0 -g -fPIC -fPIE -shared $^ -o $@

clean:
	rm -f *.o *.so demo_server foo_server
