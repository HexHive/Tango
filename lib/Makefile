CC=clang
CFLAGS=-fsanitize-coverage=edge,trace-pc-guard -O0 -g
LDFLAGS=-lrt -O0 -g -Wl,-wrap,bind

demo_server: bind_wrapper.o demo_server.o refuzz.o

clean:
	rm *.o demo_server
