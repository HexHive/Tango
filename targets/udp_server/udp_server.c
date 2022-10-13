#include <stdio.h>
#include <stdlib.h>

#include <netdb.h>
#include <netinet/in.h>

#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include <errno.h>

void bzero(void *a, size_t n) {
    memset(a, 0, n);
}

void bcopy(const void *src, void *dest, size_t n) {
    memmove(dest, src, n);
}

struct sockaddr_in* init_sockaddr_in(uint16_t port_number) {
    struct sockaddr_in *socket_address = malloc(sizeof(struct sockaddr_in));
    memset(socket_address, 0, sizeof(*socket_address));
    socket_address -> sin_family = AF_INET;
    socket_address -> sin_addr.s_addr = htonl(INADDR_ANY);
    socket_address -> sin_port = htons(port_number);
    return socket_address;
}

char* process_operation(char *input, size_t size) {
    size_t n = strnlen(input, size) * sizeof(char);
    char *output = malloc(n);
    memcpy(output, input, n);
    output[n - 1] = '\0';
    return output;
}

int main( int argc, char *argv[] ) {
    const uint16_t port_number = 5001;
    int server_fd = socket(AF_INET, SOCK_DGRAM, 0);

    struct sockaddr_in *server_sockaddr = init_sockaddr_in(port_number);
    socklen_t server_socklen = sizeof(*server_sockaddr);

    if (bind(server_fd, (const struct sockaddr *) server_sockaddr, server_socklen) < 0) {
        printf("Error! Bind has failed: %s\n", strerror(errno));
        free(server_sockaddr);
        exit(-1);
    }

    free(server_sockaddr);

    const size_t buffer_len = 256;
    char *buffer = malloc(buffer_len * sizeof(char));
    char *response = NULL;

    struct sockaddr_in client_address;
    socklen_t client_address_len = 0;

    while (1) {
        // read content into buffer from an incoming client
        int recv_len = recvfrom(server_fd, buffer, buffer_len, 0,
                           (struct sockaddr *)&client_address,
                           &client_address_len);

        if (recv_len < 0) continue;
        buffer[recv_len] = '\0';

        if (strncmp(buffer, "close", 5) == 0) {
            printf("Process %d: ", getpid());
            close(server_fd);
            printf("Closing session with `%d`. Bye!\n", server_fd);
            break;
        }
        else if (buffer[0] == '\xAA' && buffer[1] == '\xBB') {
            // CRASH!
            printf("CRASHING NOW!\n");
            fflush(stdout);
            *(volatile int *)(0) = 0x41414141;
        }

        if (strnlen(buffer, buffer_len) == 0) {
            break;
        }

        response = process_operation(buffer, buffer_len);
        bzero(buffer, buffer_len * sizeof(char));

        sendto(server_fd, response, strlen(response), 0,
            (struct sockaddr *)&client_address, sizeof(client_address));
        free(response);
    }

    free(buffer);
}
