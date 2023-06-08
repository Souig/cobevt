import torch
pthfile = r'/home/ubuntu/OpenCOOD1/opencood/logs/fax_2023_03_31_18_09_27/net_epoch16.pth'
net = torch.load(pthfile)
print(net)