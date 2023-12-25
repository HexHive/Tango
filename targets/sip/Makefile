.PHONY: kamailio

export CC_EXTRA_OPTS=${CFLAGS}
export LD_EXTRA_OPTS=${LDFLAGS} ${LDLIBS}

kamailio:
	sudo apt-get install -y bison flex
	rm -rf kamailio_repo
	git clone https://github.com/kamailio/kamailio.git kamailio_repo
	cd kamailio_repo && \
	git checkout 2648eb3 && \
	patch -p1 < ../kamailio.patch && \
	$(MAKE) MEMPKG=sys cfg && \
	$(MAKE) all -j20 && \
	cp -r ./src/modules ../ && \
	cp ./src/kamailio ../
