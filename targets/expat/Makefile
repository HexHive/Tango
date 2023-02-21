.PHONY: libexpat

export LDFLAGS := $(LDFLAGS) $(LDLIBS)

libexpat:
	rm -rf libexpat_repo
	git clone https://github.com/libexpat/libexpat.git libexpat_repo
	cd libexpat_repo && \
	patch -p1 < ../hashsalt.patch && \
	patch -p1 < ../bufsize.patch && \
	cd expat && \
	mkdir build && cd build && \
	cmake \
		-DEXPAT_BUILD_TOOLS=ON \
		-DEXPAT_BUILD_EXAMPLES=OFF \
		-DEXPAT_BUILD_TESTS=OFF \
		-DEXPAT_SHARED_LIBS=OFF \
		-DEXPAT_BUILD_DOCS=OFF \
		-DEXPAT_BUILD_PKGCONFIG=OFF \
		-DEXPAT_DEV_URANDOM=OFF \
		.. && \
	$(MAKE) && \
	cp xmlwf/xmlwf ../../../xmlwf
