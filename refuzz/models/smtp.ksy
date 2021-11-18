meta:
  id: smtp
  endian: le
  encoding: ascii

variables:
  extended_commands:
    type: strz[]
    value: '[]'
  has_greeted:
    value: false
  last_sent_command:
    type: strz
    value: '""'
  client_address:
    type: strz
    value: _global.get_ip_address()

states:
  entry:
    initial: handshake

inputs:
  handshake:
    seq:
      - id: server_greet
        type: server_response
        action: receive
      - id: client_ehlo
        type: client_command
        action: send
        constraints:
          _this.command: '"EHLO"'
      - id: server_ehlo
        type: server_response
        action: receive

types:
  empty:
    seq:
      - size: 0

  crlf:
    seq:
      - id: newline
        contents: "\r\n"

  server_response:
    seq:
      - id: lines
        type: response_line
        repeat: until
        repeat-until: _.begin.separator == ' '
      - type: empty
        exports:
          _root.last_sent_command: '""' # FIXME might need to disable this with "chunking"

    types:
      response_line:
        seq:
          - id: begin
            type: begin
          - id: body
            doc: |
              Because of stuff (TM), the body must terminate itself with CRLF
            type:
              switch-on: _root.last_sent_command
              cases:
                '"EHLO"': ehlo_ok_rsp
                _: raw_body
          # - id: end
          #   type: end

        types:
          ## Standard response begin and end
          begin:
            seq:
              - id: code
                size: 3
              - id: separator
                type: str
                size: 1
                choices:
                  - '" "'
                  - '"-"'
          end:
            seq:
              - type: crlf

          ## Different response bodies depending on command sent
          # Raw unparsed message
          raw_body:
            seq:
              - id: data
                type: str
                consume: false
                terminator: 13
              - type: crlf

          # ehlo-ok-rsp
          ehlo_ok_rsp:
            seq:
              - id: msg
                type:
                  switch-on: _root.has_greeted
                  cases:
                    false: ehlo_greet
                    true: ehlo_line
                constraints:
                  _parent.begin.code: '[50, 53, 48]' # 250, but needs to be expressed like this cuz CalcStrType !~= BytesType apparently

            types:
              keyword:
                seq:
                  - id: c
                    size: 1
                    repeat: until
                    repeat-until: c.last[0] == 32 or c.last[0] == 13
                  - id: lf
                    if: terminator == 13
                    contents: "\n" # this constructs the CRLF by appending LF to the consumed CR terminator
                  - id: params
                    if: terminator == 32
                    type: str
                    terminator: 13
                    consume: false
                  - id: newline
                    if: terminator == 32
                    type: crlf
                instances:
                  name:
                    value: c.to_b.to_s('ascii').substring(0, -1)
                  terminator:
                    value: c.last[0]

              ehlo_greet:
                seq:
                  - id: domain
                    type: keyword
                    exports:
                      _root.has_greeted: true

              ehlo_line:
                seq:
                  - id: keyword
                    type: keyword
                    exports:
                      _root.extended_commands: _root.extended_commands + [keyword.name]
  # server_response

  client_command:
    seq:
      - id: command
        type: str
        terminator: 13 # this is a hack to silence KSC; we will never parse a command (for now); we will only generate it
        consume: false
        choices:
          - '"EHLO"'
          - '"MAIL"'
        exports:
          _root.last_sent_command: command
      - id: params
        type:
          switch-on: command
          cases:
            '"EHLO"': ehlo_cmd
            '"MAIL"': mail_cmd
            _: empty
      - id: newline
        contents: "\r\n"

    types:
      ehlo_cmd:
        seq:
          - id: separator
            contents: " "
          - id: addr
            type: str
            terminator: 13
            consume: false
            valid: _root.client_address

      mail_cmd:
        seq:
          - id: separator
            contents: " "
          - id: frm
            type: str
            valid: '"FROM"'
            terminator: 58 # ':'
