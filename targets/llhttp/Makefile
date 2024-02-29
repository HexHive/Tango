.PHONY: llhttp

export CLANG := $(CC)

llhttp:
	rm -rf llhttp_repo
	git clone https://github.com/nodejs/llhttp.git llhttp_repo
	cd llhttp_repo && \
	git checkout a35e183 && \
	npm install && \
	$(MAKE) build/libllhttp.a && \
	cd - && \
	$(CC) $(CFLAGS) -c driver.c -o driver.o -I llhttp_repo/build/ && \
	$(CC) $(CFLAGS) driver.o llhttp_repo/build/libllhttp.a \
		-o parse $(LDFLAGS) $(LDLIBS)
