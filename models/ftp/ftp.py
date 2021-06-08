import socket
import threading
import sys
import logging
import pwd
import crypt
import os
import pam
from hmac import compare_digest 
from enum import Enum

class FTPState(Enum):
    START = 0
    INIT = 220
    UNAME_CORRECT = 220
    UNAME_WRONG = 500
    PASS_CORRECT = 230
    PASS_WRONG = 501


status_codes = {
    'INIT': (220, 'Server ready\n'),
    'UNAME_CORRECT': (331, 'Password Required for the User\n'),
    'UNAME_WRONG': (500, 'Username provided is wrong\n'),
    'PASS_CORRECT': (230, 'User authenticated\n'),
    'PASS_WRONG': (501, 'Password incorrect\n'),
    'PATHNAME_CREATED': (257, 'Directory Created successfully\n')
}

class FTPConnection(threading.Thread):
    def __init__(self, conn: socket.socket, addr):
        threading.Thread.__init__(self)

        self.conn = conn
        self.addr = addr

        self.logger = logging.getLogger(f'FTP Connection {addr}')
        self.current_state = FTPState.START
        self.current_user = ''
        self.authenticated = False
        self.pam = pam.pam()
        self.current_wd = os.getcwd()
    
    def run(self):
        self.conn.sendall(bytes(status_codes['INIT'][1], encoding='utf-8'))

        print('INIT SENT SUCCESSFULLY')

        self.current_state = FTPState.INIT

        while True:

            command = self.recieve_command(256)
            if not command:
                break
            self.handle_command(command)
            print(command)

        self.conn.close()

    def send_response(self, message):
        self.conn.sendall(bytes(message, encoding='utf-8'))

    def recieve_command(self, length: int):
        command = self.conn.recv(length).decode('utf-8')
        return command

    def handle_command(self, command):
        command = command.strip('\n')

        if 'USER' in command:

            # Grab the username and check if this user exists on the system
            username = command.split(' ')[1]
            try:
                pwd.getpwnam(username)
                self.current_state = FTPState.UNAME_CORRECT
                self.current_user = username
                self.send_response(status_codes['UNAME_CORRECT'][1])
            except KeyError:
                self.current_state = FTPState.UNAME_WRONG
                self.send_response(status_codes['UNAME_WRONG'][1])
        elif 'PASS' in command:
            if self.current_state == FTPState.UNAME_CORRECT:
                pass_given = command.split(' ')[1]
                if self.pam.authenticate(self.current_user, pass_given):
                    self.current_state = FTPState.PASS_CORRECT
                    self.authenticated = True
                    self.send_response(status_codes['PASS_CORRECT'][1])
                else:
                    self.send_response(status_codes['PASS_WRONG'][1])

            else:
                self.send_response("Plese send a correct username with USER command")
        elif 'LIST' in command:
            if self.authenticated:
                filelist = ''.join([file + '\n' for file in os.listdir(self.current_wd)])
                self.send_response(filelist)
            else:
                self.send_response("User not authenticated")
        elif 'PUT' in command:
            # requires an ftp client on the other side
            pass
        elif 'MKDIR' in command:
            dir_name = command.split(' ')[1]
            if self.authenticated:
                os.mkdir(os.path.join(self.current_wd,dir_name))
                self.send_response(status_codes['PATHNAME_CREATED'][1])
        elif 'BYE' in command:
            self.conn.shutdown(socket.SHUT_RDWR)
            self.conn.close()

class FTPServer():

    def __init__(self, host: str, port: int):
        threading.Thread.__init__(self)
        self.host = host
        self.port = port
        self.conn_list = list()
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serversocket.bind((host, port))

        self.logger = logging.getLogger('FTPServer')
        self.logger.setLevel(-1)
        print("Address bound successfully")
        

    def run(self):
        try:
            while True:
                self.serversocket.listen(5)

                conn_socket, conn_addr = self.serversocket.accept()

                print(f"Connection created successfully with {conn_addr}")

                ftp_conn = FTPConnection(conn_socket, conn_addr)
                ftp_conn.start()

                self.conn_list.append(ftp_conn)

        except KeyboardInterrupt:

            self.serversocket.close()
            print("Closing the server")

if __name__ == '__main__':
    ftp = FTPServer('', int(sys.argv[1]))

    ftp.run()
