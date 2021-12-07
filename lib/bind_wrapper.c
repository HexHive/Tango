/*
 Copyright (c) 2021 Qiang Liu <cyruscyliu@gmail.com>

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/prctl.h>

static struct linger no_linger = {0};
static uint32_t reuse = 1;

int __wrap_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
int __real_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);

pid_t __wrap_fork();
pid_t __real_fork();

/* Refer to this SO answer on the nitty-gritty about TIME_WAIT
 * https://stackoverflow.com/a/14388707
 */
int __wrap_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    setsockopt(sockfd, SOL_SOCKET, SO_LINGER, &no_linger, sizeof(no_linger));
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse));
    printf("Enable address-port reuse on socket: %d\n", sockfd);

    /* go to bind then */
    return __real_bind(sockfd, addr, addrlen);
}

pid_t __wrap_fork() {
    pid_t ppid = getpid();
    pid_t child_pid = __real_fork();
    if (child_pid == 0) {
        prctl(PR_SET_PDEATHSIG, SIGTERM);
        if (getppid() != ppid)
            exit(-256);
    }
    return child_pid;
}
