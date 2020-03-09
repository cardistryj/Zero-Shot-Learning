import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

#SRNN
class SRNN(nn.Module):
    def __init__(self,num_class,dim):
        super(SRNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (3,3),stride=1,padding=(1,1))
        self.conv2 = nn.Conv2d(64, 128, (3,3),stride=1,padding=(1,1))
        self.conv3 = nn.Conv2d(128, 256, (3,3),stride=1,padding=(1,1))
        self.conv4 = nn.Conv2d(256, 512, (3,3),stride=1,padding=(1,1))

        self.maxPool = nn.MaxPool2d((1,2),stride=(1,2),return_indices=True)
        self.avgPool = nn.AvgPool2d((2,2),stride=2)

        self.fc0 = nn.Linear(64*512, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, dim)
        self.fc3 = nn.Linear(dim, num_class)  

        self.dropoutConv1 = nn.Dropout2d()  
        self.dropoutConv2 = nn.Dropout2d()
        self.dropoutConv3 = nn.Dropout2d()
        self.dropoutConv4 = nn.Dropout2d()  
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.dropout3 = nn.Dropout()

        self.bn1=nn.BatchNorm1d(1024)
        self.bn2=nn.BatchNorm1d(512)
        self.bn3=nn.BatchNorm1d(dim)

        self.bnconv1=nn.BatchNorm2d(64)
        self.bnconv2=nn.BatchNorm2d(128)
        self.bnconv3=nn.BatchNorm2d(256)
        self.bnconv4=nn.BatchNorm2d(512)

        ###############
        # autoencoder
        self.conv1_r = nn.ConvTranspose2d(64, 1, (3,3),stride=1,padding=(1,1))
        self.conv2_r = nn.ConvTranspose2d(128, 64, (3,3),stride=1,padding=(1,1))
        self.conv3_r = nn.ConvTranspose2d(256, 128, (3,3),stride=1,padding=(1,1))
        self.conv4_r = nn.ConvTranspose2d(512, 256, (3,3),stride=1,padding=(1,1))

        self.maxPool_r = nn.MaxUnpool2d((1,2),stride=(1,2))
        self.avgPool_r = nn.UpsamplingNearest2d(scale_factor=2)

        self.fc0_r = nn.Linear(1024, 64*512)
        self.fc1_r = nn.Linear(512, 1024)
        self.fc2_r = nn.Linear(dim, 512)

        self.dropoutConv1_r = nn.Dropout2d()  
        self.dropoutConv2_r = nn.Dropout2d()
        self.dropoutConv3_r = nn.Dropout2d()
        self.dropoutConv4_r = nn.Dropout2d()  
        self.dropout1_r = nn.Dropout()
        self.dropout2_r = nn.Dropout()
        self.dropout3_r = nn.Dropout()

        self.bn1_r=nn.BatchNorm1d(64*512)
        self.bn2_r=nn.BatchNorm1d(1024)
        self.bn3_r=nn.BatchNorm1d(512)

        self.bnconv1_r=nn.BatchNorm2d(1)
        self.bnconv2_r=nn.BatchNorm2d(64)
        self.bnconv3_r=nn.BatchNorm2d(128)
        self.bnconv4_r=nn.BatchNorm2d(256)
        ##############


    def forward(self, x):
        x = x.view(-1,1,2,1024)
        
        x, indices1 = self.maxPool(self.dropoutConv1(F.relu(self.bnconv1(self.conv1(x)))))
        x, indices2 = self.maxPool(self.dropoutConv2(F.relu(self.bnconv2(self.conv2(x)))))
        x, indices3 = self.maxPool(self.dropoutConv3(F.relu(self.bnconv3(self.conv3(x)))))
        x = self.avgPool(self.dropoutConv4(F.relu(self.bnconv4(self.conv4(x)))))

        x = x.view(-1, 64*512)
        x = self.dropout1(F.relu(self.bn1(self.fc0(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc1(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc2(x))))
        
        x = self.fc3(x)
        return x

    def decoder(self, x):
        x = x.view(-1,1,2,1024)
        
        x, indices1 = self.maxPool(self.dropoutConv1(F.relu(self.bnconv1(self.conv1(x)))))
        x, indices2 = self.maxPool(self.dropoutConv2(F.relu(self.bnconv2(self.conv2(x)))))
        x, indices3 = self.maxPool(self.dropoutConv3(F.relu(self.bnconv3(self.conv3(x)))))
        x = self.avgPool(self.dropoutConv4(F.relu(self.bnconv4(self.conv4(x)))))

        x = x.view(-1, 64*512)
        x = self.dropout1(F.relu(self.bn1(self.fc0(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc1(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc2(x))))
        
        x = F.relu(self.bn3_r(self.fc2_r(x)))
        x = F.relu(self.bn2_r(self.fc1_r(x)))
        x = F.relu(self.bn1_r(self.fc0_r(x)))

        x = x.view(-1,512,1,64)

        x = F.relu(self.bnconv4_r(self.conv4_r(self.avgPool_r(x))))
        x = F.relu(self.bnconv3_r(self.conv3_r(self.maxPool_r(x,indices3))))
        x = F.relu(self.bnconv2_r(self.conv2_r(self.maxPool_r(x,indices2))))
        x = F.relu(self.bnconv1_r(self.conv1_r(self.maxPool_r(x,indices1))))

        x=x.view(-1,2,1024)

        return x

    #获得语义向量
    def getSemantic(self, x):        
        x = x.view(-1,1,2,1024)
        
        x, indices1 = self.maxPool(self.dropoutConv1(F.relu(self.bnconv1(self.conv1(x)))))
        x, indices2 = self.maxPool(self.dropoutConv2(F.relu(self.bnconv2(self.conv2(x)))))
        x, indices3 = self.maxPool(self.dropoutConv3(F.relu(self.bnconv3(self.conv3(x)))))
        x = self.avgPool(self.dropoutConv4(F.relu(self.bnconv4(self.conv4(x)))))

        x = x.view(-1, 64*512)
        x = self.dropout1(F.relu(self.bn1(self.fc0(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc1(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc2(x))))

        return x

#构造18层的Signal Resnet
def getSRNN(num_class,dim):
    model = SRNN(num_class,dim)

    return model
