import socket
from threading import Thread
import numpy as np
import torch


class Server():
    def __init__(self,max_client,shape):
        self.server = socket.socket()
        self.ip = "127.0.0.1"
        self.port = 2000
        addr = (self.ip,self.port)
        self.server.bind(addr)
        self.server.settimeout(300)
        self.max_clients = max_client
        self.recv_num = 0
        self.messages = []
        self.shape = shape
        self.server.listen(self.max_clients)
        self.start_listen()
        

    def start_listen(self):
        while(self.recv_num < self.max_clients):
            client,addr = self.server.accept()
            leng = client.recv(1)
            if leng.decode() == str(self.recv_num+1):
                self.recv_num += 1
                print("('"+addr[0]+"',"+str(addr[1])+")已成功连接")
                len = client.send("ACK".encode())
                self.recv_msg(client)
    
    def recv_msg(self,client):
        
        y = []
        for i in range(len(self.shape)):
            print(self.shape[i])
            b,l,m,h,w,c = self.shape[i]
            length = b*l*m*h*w*c
            times = round(4*length/16384)
            for j in range(times):
                msg1 = client.recv(16384)
                if len(msg1)!=16384:
                    print(j, len(msg1))
                
                if j == 0:
                    msg = msg1
                else:
                    msg = msg + msg1
                if j == times - 1 and len(msg) < 4*length:
                    msg += client.recv(4*length-len(msg))
  
            msg = np.frombuffer(msg,dtype=np.float32)
            print("Successfully received.")  
            print(msg.shape)
            #msg1 = msg[i*b*l*m*h*w*c:(i+1)*b*l*m*h*w*c]
            msg = torch.from_numpy(msg).view(b,l,m,h,w,c)
            print(msg.shape)
            y.append(msg.cuda())
        self.messages.append(y)

if __name__ == '__main__':
    shape = [(2,3),(2,3),(2,3)]
    server = Server(5,shape)
