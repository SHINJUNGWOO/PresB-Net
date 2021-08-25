import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from proposed_models import  Aresblock1_6 

class pre_conv(nn.Module):
#{{{
    def __init__(self, in_channel, out_channel, stride):
        super(pre_conv, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)

        return x
#}}}

class proposed(nn.Module):    
    def __init__(self, stage_channel = (64,64,64,128,128,256,256,512,512),binarized = False ,num_class = 100):
        
        stage_channel = [128] + [2*i for i in stage_channel[1:]]

        super(proposed,self).__init__()
        self.feature = nn.ModuleList()
        self.num_class = num_class
        for i,out_channel in enumerate(stage_channel):
            if i==0:
                self.feature.append(pre_conv(3,out_channel,1))
                #self.feature.append(nn.BatchNorm2d(out_channel))
                #self.feature.append(nn.ReLU())
                #self.feature.append(nn.BatchNorm2d(out_channel))
            #elif(stage_channel[i-1] != out_channel and out_channel !=64):
            #    self.feature.append(Aresblock(stage_channel[i - 1], out_channel, 2))
            elif(stage_channel[i-1] != out_channel):
                self.feature.append(Aresblock1_6(stage_channel[i - 1], out_channel, stride=2,binarized = binarized))
            else:
                self.feature.append(Aresblock1_6(stage_channel[i - 1], out_channel, stride=1,binarized = binarized))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_channel[-1],num_class)
        self.soft = nn.Softmax(dim=1)



    def forward(self,x):
        for block in self.feature:
            x = block(x)

        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        #x = self.soft(x)

        return x


