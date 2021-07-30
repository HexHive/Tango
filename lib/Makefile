CC=clang
CFLAGS=-fsanitize-coverage=func,trace-pc-guard -O0 -g
LDFLAGS=-lrt -O0 -g

demo_server: demo_server.o refuzz.o
