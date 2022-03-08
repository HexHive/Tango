#include <stdio.h>
#include <stdlib.h>

#include <netdb.h>
#include <netinet/in.h>

#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>

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

char* process_operation(char *input) {
    size_t n = strlen(input) * sizeof(char);
    char *output = malloc(n);
    memcpy(output, input, n);
    return output;
}

int main( int argc, char *argv[] ) {
    const uint16_t port_number = 5001;
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in *server_sockaddr = init_sockaddr_in(port_number);
    struct sockaddr_in *client_sockaddr = malloc(sizeof(struct sockaddr_in));
    socklen_t server_socklen = sizeof(*server_sockaddr);
    socklen_t client_socklen = sizeof(*client_sockaddr);


    if (bind(server_fd, (const struct sockaddr *) server_sockaddr, server_socklen) < 0)
    {
        printf("Error! Bind has failed\n");
        exit(0);
    }
    if (listen(server_fd, 3) < 0)
    {
        printf("Error! Can't listen\n");
        exit(0);
    }


    const size_t buffer_len = 256;
    char *buffer = malloc(buffer_len * sizeof(char));
    char *response = NULL;
    __pid_t pid = -1;

    int client_fd = accept(server_fd, (struct sockaddr *) &client_sockaddr, &client_socklen);

    if (client_fd == -1) {
        exit(0);
    }

    printf("Connection with `%d` has been established.\nWaiting for a query...\n", client_fd);

    while (1) {
        int recv_len = recv(client_fd, buffer, buffer_len, 0);

        if (strncmp(buffer, "close", 5) == 0) {
            printf("Process %d: ", getpid());
            close(client_fd);
            printf("Closing session with `%d`. Bye!\n", client_fd);
            break;
        }
        else if (buffer[0] == '\xAA' && buffer[1] == '\xBB') {
            // CRASH!
            printf("CRASHING NOW!\n");
            fflush(stdout);
            *(volatile int *)(0) = 0x41414141;
        }

        if (strlen(buffer) == 0) {
            break;
        }

        free(response);
        response = process_operation(buffer);
        bzero(buffer, buffer_len * sizeof(char));

        send(client_fd, response, strlen(response), 0);
    }
}
