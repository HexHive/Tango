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

COPY git_install.sh /home/tango/git_install.sh
RUN cd $HOME/ && ./git_install.sh -y

WORKDIR /home/tango
ENTRYPOINT ["scripts_eurosp/docker_entry.sh"]
