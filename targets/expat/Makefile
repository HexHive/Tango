.PHONY: libexpat

export CFLAGS := $(CFLAGS) -D_DEBUG
export LDFLAGS := $(LDFLAGS) $(LDLIBS)

libexpat:
	rm -rf libexpat_repo
	git clone https://github.com/libexpat/libexpat.git libexpat_repo
	cd libexpat_repo/expat && \
	./buildconf.sh && \
	./configure \
		--without-examples \
		--without-tests \
		--without-docbook \
		--with-pic && \
	$(MAKE) && \
	cp xmlwf/.libs/xmlwf ../../xmlwf
