.PHONY: yajl

export LDFLAGS := $(LDFLAGS) $(LDLIBS)

yajl:
	rm -rf yajl_repo
	git clone https://github.com/lloyd/yajl.git yajl_repo
	cd yajl_repo && \
	mkdir build && cd build && \
	cmake .. && \
	$(MAKE) && \
	cp test/parsing/yajl_test ../../parse
