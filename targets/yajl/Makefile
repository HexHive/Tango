.PHONY: yajl

export LDFLAGS := $(LDFLAGS) $(LDLIBS)

yajl:
	rm -rf yajl_repo
	git clone https://github.com/openEuler-BaseService/yajl.git yajl_repo
	cd yajl_repo && \
	git checkout 175a30f && \
	mkdir build && cd build && \
	cmake .. && \
	$(MAKE) && \
	cp test/parsing/yajl_test ../../parse
