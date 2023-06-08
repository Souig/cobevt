import socket
from threading import Thread
import numpy as np
import torch

BUFFER_SIZE = 1024

class Client():
    def __init__(self,msg,num):
        self.client = socket.socket()
        self.ip = "127.0.0.1"
        self.port = 2000
        self.msg = msg
        self.num = num
        self.start_connection()

    def start_connection(self):
        m_address = (self.ip,self.port)
        self.client.connect(m_address)
        print("Successfully connected.")
        leng = self.client.send(str(self.num).encode())
        print(leng,str(self.num).encode())
        msg = self.client.recv(BUFFER_SIZE).decode()
        if msg == 'ACK':
            self.send_msg()

    def send_msg(self):
        for i in range(len(self.msg)):
            print(self.msg[i].shape)
            self.msg[i] = (self.msg[i].cpu().numpy().astype(np.float32)).tobytes()
            length = len(self.msg[i])
            print(length)
            times = round(length/16384)
            for j in range(times):
                if j == times - 1:
                    leng = self.client.send(self.msg[i][j*16384:])
                else:
                    leng = self.client.send(self.msg[i][j*16384:(j+1)*16384])
                
            #leng = self.client.send(self.msg[i])
            print("Message of %d length has been sent." %(length) )
        
        self.client.close()

if __name__ == '__main__':
    y = []
    y1 = torch.tensor([[1,1,1],[1,1,1]],dtype=torch.float32)
    y2 = torch.tensor([[2,2,2],[2,2,2]],dtype=torch.float32)
    y3 = torch.tensor([[3,3,3],[3,3,3]],dtype=torch.float32)
    y.append(y1)
    y.append(y2)
    y.append(y3)

    client = Client(y)