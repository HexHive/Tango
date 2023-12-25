FROM ubuntu:22.04

ENV TZ="Europe/Zurich"
RUN apt-get update && apt-get install -y git sudo software-properties-common

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN mkdir -p /home && \
	groupadd -g ${GROUP_ID} tango && \
	useradd -l -u ${USER_ID} -K UMASK=0000 -d /home -g tango tango && \
	chown tango:tango /home
RUN	echo "tango:ognat" | chpasswd && usermod -a -G sudo tango
RUN echo "tango ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers

USER tango:tango
ENV PATH="/home/.local/bin:$PATH"
RUN mkdir -p $HOME/.ssh
ADD --chown=tango:tango id_deploy /home/.ssh/id_ed25519
RUN chmod 400 /home/.ssh/id_ed25519
RUN touch $HOME/.ssh/known_hosts && \
	ssh-keyscan github.com >> $HOME/.ssh/known_hosts
RUN git clone git@github.com:HexHive/tangofuzz.git $HOME/tango && \
	cd $HOME/tango && \
	git submodule init && \
	git submodule update scripts/

COPY git_install.sh /home/tango/scripts/git_install.sh
RUN cd $HOME/tango/scripts && ./git_install.sh -y

WORKDIR /home/tango
ENTRYPOINT ["scripts/docker_entry.sh"]
