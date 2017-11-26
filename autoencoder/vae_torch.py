import datetime
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(asctime)s:%(message)s)")
# if not os.path.exists('./vae_img'):
#    os.mkdir('vae_img')

def to_img(x):
    x=x.clamp(0,1)
    # x=x.view(x.size(0),3,32,32)
    return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 3*32*32 -> 32 * 32 *32
        self.conv1=nn.Sequential(nn.Conv2d(3, 32, 5, stride=1,padding=2), nn.BatchNorm2d(32),\
                nn.ReLU(),\
                nn.MaxPool2d(2,return_indices=True)) 
        # 32*16*16 -> 64*16*16
        self.conv2=nn.Sequential(nn.Conv2d(32, 64, 3, stride=1,padding=1), nn.BatchNorm2d(64),\
                nn.ReLU())
        # 64*16*16 -> 64*16*16
        self.conv3=nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(64*16*16, 1024),nn.BatchNorm1d(1024), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(1024, 100),nn.BatchNorm1d(100), nn.ReLU())
        self.fc22 = nn.Sequential(nn.Linear(1024, 100), nn.BatchNorm1d(100), nn.ReLU())
    
    def forward(self, img):
        img, indices = self.conv1(img)
        # print("indice:",indices)
        img = self.conv2(img)
        img = self.conv3(img)
        res = img.view(-1 ,64*16*16)
        res = self.fc1(res)
        mu, sigma = self.fc21(res), self.fc22(res)
        return mu, sigma, indices

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(100, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1024, 64*16*16), nn.BatchNorm1d(64*16*16), nn.ReLU())
        self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(64, 64, 3, padding = 1), nn.BatchNorm2d(64),
                nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3,padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(32, 3, 5,padding=2), nn.BatchNorm2d(3), nn.ReLU())
        self.unpool = nn.MaxUnpool2d(2)
        self.conv4 = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3), nn.Sigmoid())

    def forward(self,x, indices):
        x = self.fc1(x)
        x = self.fc2(x)
        img = x.view(-1, 64, 16, 16)
        img = self.deconv1(img)
        img = self.deconv2(img)
        # print(img.size(), indices.size())
        img = self.unpool(img, indices=indices)
        img = self.deconv3(img)
        img = self.conv4(img)
        # print(img.size())
        return img

class VaeAutoencoder(nn.Module):
    def __init__(self):
        super(VaeAutoencoder, self).__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()
    def reparameterize(self, mu, logvar):
        '''
        mu：均值
        logvar：log(方差) 防止过大溢出
        '''
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(mu.size()).normal_()
        else:
            eps = torch.FloatTensor(mu.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, img):
        mu, logvar,indices = self.encoder(img)
        x = self.reparameterize(mu, logvar)
        res = self.decoder(x,indices)
        return res, mu, logvar
    
recst_loss = nn.MSELoss()
def loss_function(recon_x, x, mu, logvar):
    ''' 
    recon_x：重建图像
    x：原图像
    mu：隐变量均值
    logvar：log(隐变量方差)
    '''
    BCE = recst_loss(recon_x, x)
    KLD = torch.sum(1+logvar-mu**2-logvar.exp_()).mul_(-0.5)
    return BCE + KLD
    
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

data_dir = "/home/mintyi/codework/data"
if __name__=="__main__":
    # convert to tensor [0,1]
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root=data_dir,train=True,download = False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 10, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_dir,train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

    # classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

    vae = VaeAutoencoder()
    vae.train() # set model into train mode
    optimizer = optim.Adam(vae.parameters(),lr=1e-3)
    if torch.cuda.is_available():
        vae.cuda()
    logging.info("start train")
    for epoch in range(num_epochs):
        train_loss = 0
        origin_img = None # 用于存原图片
        for batchidx, data in enumerate(trainloader):
            img, _ = data
            origin_img = img
            img = Variable(img)
            if torch.cuda.is_available():
                img = img.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(img)
            loss = loss_function(recon_batch, img, mu, logvar)
            logging.debug("Loss is %.6f",loss.data[0])
            loss.backward()
            train_loss+=loss.data[0]
            optimizer.step()

            if batchidx%10==0:
                logging.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batchidx*len(img),len(trainloader.dataset), 10*batchidx/len(trainloader),loss.data[0]/len(img)))

            # if batchidx==10:
            #     break
        logging.info("====> Epoch: {} Average loss: {:.4f}".format(epoch,train_loss/len(trainloader.dataset)))
        if epoch%10==0:
            save = to_img(recon_batch.cpu().data)
            # print(save.size(), origin_img.size(), type(save), type(origin_img))
            save = torch.cat((save, origin_img))
            save_image(save, "/home/mintyi/codework/vae_img/image_{}.png".format(epoch))
        break
    # torch.save(vae.state_dict(),'./vae_cifar.model')
