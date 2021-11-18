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

#include <stdio.h>
#include <stdint.h>
#include <sys/socket.h>

int __wrap_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
int __real_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);

int __wrap_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    uint32_t optval;
    uint32_t optlen;

    /* get and set SO_REUSEADDR */
    getsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, &optlen);
    if (optval != 0) {
        printf("Socket: %d has enabled SO_REUSEADDR\n", sockfd);
    } else {
        optval = 1;
        setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
        printf("Enable SO_REUSEADDR on socket: %d\n", sockfd);
    }

    /* go to bind then */
    return __real_bind(sockfd, addr, addrlen);
}
