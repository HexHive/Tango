#include "llhttp.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

// Input buffer.
#define MaxInputChunkSize 4
static uint8_t InputBuf[MaxInputChunkSize];

int handle_on_message_complete(llhttp_t *arg) { return 0; }

int main(int argc, char **argv) {
  uint8_t config[4];
  ssize_t cur_read, n_read = 0;
  while ((cur_read = read(0, &InputBuf[n_read], MaxInputChunkSize - n_read),
      n_read += cur_read, cur_read) > 0 && n_read < sizeof(config));
  if (n_read >= sizeof(config)) {
    n_read -= sizeof(config);
    memmove(config, InputBuf, sizeof(config));
    memcpy(InputBuf, &InputBuf[sizeof(config)], n_read);
  }
  else {
    fputs("Not enough bytes available for config!", stderr);
    return -1;
  }

  int headers = (config[0] & 0x01) == 1;
  int chunked_length = (config[1] & 0x01) == 1;
  int keep_alive = (config[2] & 0x01) == 1;
  llhttp_type_t http_type;
  if (config[3] % 3 == 0) {
    http_type = HTTP_BOTH;
  } else if (config[3] % 3 == 1) {
    http_type = HTTP_REQUEST;
  } else {
    http_type = HTTP_RESPONSE;
  }

  llhttp_t parser;
  llhttp_settings_t settings;

  /* Initialize user callbacks and settings */
  llhttp_settings_init(&settings);

  /* Set user callback */
  settings.on_message_complete = handle_on_message_complete;

  llhttp_init(&parser, http_type, &settings);
  llhttp_set_lenient_headers(&parser, headers);
  llhttp_set_lenient_chunked_length(&parser, chunked_length);
  llhttp_set_lenient_keep_alive(&parser, keep_alive);

  cur_read = n_read;
  do {
    if (llhttp_execute(&parser, (const char*)InputBuf, cur_read) != HPE_OK) {
      fprintf(stderr, "Failed after %zi bytes.\n", n_read);
      return -1;
    }
  } while ((cur_read = read(0, InputBuf, MaxInputChunkSize),
      n_read += cur_read, cur_read) > 0);
  fprintf(stderr, "Successfully processed %zi bytes.\n", n_read);
  return 0;
}
