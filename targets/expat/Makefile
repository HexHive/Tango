.PHONY: libexpat

export CFLAGS := $(CFLAGS)
export LDFLAGS := $(LDFLAGS) $(LDLIBS)

libexpat:
	rm -rf libexpat_repo
	git clone https://github.com/libexpat/libexpat.git libexpat_repo
	cd libexpat_repo && \
	patch -p1 < ../hashsalt.patch && \
	patch -p1 < ../bufsize.patch && \
	cd expat && \
	./buildconf.sh && \
	./configure \
		--without-examples \
		--without-tests \
		--without-docbook \
		--with-pic && \
	$(MAKE) && \
	cp xmlwf/.libs/xmlwf ../../xmlwf
