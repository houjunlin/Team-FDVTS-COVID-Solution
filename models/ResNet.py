from __future__ import absolute_import

import torch
import math
import copy
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F

from models import inflate
from models import AP3D
from models import NonLocal
from models import resnet1

# import inflate
# import AP3D
# import NonLocal
# import resnet_2d

from models.resnet3D import resnet50 #medical
from models.resnest import resnest50_3D


__all__ = ['AP3DResNet50', 'AP3DNLResNet50']


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)      


class Bottleneck3D(nn.Module):
    def __init__(self, bottleneck2d, block, inflate_time=False, temperature=4, contrastive_att=True):
        super(Bottleneck3D, self).__init__()

        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        if inflate_time == True:
            self.conv2 = block(bottleneck2d.conv2, temperature=temperature, contrastive_att=contrastive_att)
        else:
            self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(
            inflate.inflate_conv(downsample2d[0], time_dim=1, 
                                 time_stride=time_stride),
            inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet503D(nn.Module):
    def __init__(self, num_classes, block, c3d_idx, nl_idx, temperature=4, contrastive_att=True, **kwargs):
        super(ResNet503D, self).__init__()

        self.block = block
        self.temperature = temperature
        self.contrastive_att = contrastive_att

        resnet2d = torchvision.models.resnet50(pretrained=True)
        # resnet2d = resnet1.resnet50(pretrained=False)
        # ckpt = torch.load('/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/cciidistpreimagenet/44.pkl')
        # state_dict = ckpt['net']
        # unParalled_state_dict = {}
        # for key in state_dict.keys():
        #     unParalled_state_dict[key.replace("module.features.", "")] = state_dict[key]
        # resnet2d.load_state_dict(unParalled_state_dict,False)

        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)     
           
        self.conv1 = inflate.inflate_conv(new_conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1, c3d_idx=c3d_idx[0], \
                                             nonlocal_idx=nl_idx[0], nonlocal_channels=256)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, c3d_idx=c3d_idx[1], \
                                             nonlocal_idx=nl_idx[1], nonlocal_channels=512)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, c3d_idx=c3d_idx[2], \
                                             nonlocal_idx=nl_idx[2], nonlocal_channels=1024)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4, c3d_idx=c3d_idx[3], \
                                             nonlocal_idx=nl_idx[3], nonlocal_channels=2048)

        # self.bn = nn.BatchNorm1d(2048)
        # self.bn.apply(weights_init_kaiming)

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Linear(2048, num_classes)
        self.classifier.apply(weights_init_classifier)

    def _inflate_reslayer(self, reslayer2d, c3d_idx, nonlocal_idx=[], nonlocal_channels=0):
        reslayers3d = []
        for i,layer2d in enumerate(reslayer2d):
            if i not in c3d_idx:
                layer3d = Bottleneck3D(layer2d, AP3D.C2D, inflate_time=False)
            else:
                layer3d = Bottleneck3D(layer2d, self.block, inflate_time=True, \
                                       temperature=self.temperature, contrastive_att=self.contrastive_att)
            reslayers3d.append(layer3d)

            if i in nonlocal_idx:
                non_local_block = NonLocal.NonLocalBlock3D(nonlocal_channels, sub_sample=True)
                reslayers3d.append(non_local_block)

        return nn.Sequential(*reslayers3d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # b,2048,2,32,32

        x = self.avgpool(x)
        features = torch.flatten(x,1)

        x = self.classifier(features)

        return features, x

def AP3DResNet50(num_classes, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D(num_classes, AP3D.APP3DC, c3d_idx, nl_idx, **kwargs)

def P3DCResNet50(num_classes, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D(num_classes, AP3D.P3DC, c3d_idx, nl_idx, **kwargs)

def C2DResNet50(num_classes, **kwargs):
    c3d_idx = [[],[],[],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D(num_classes, AP3D.C2D, c3d_idx, nl_idx, **kwargs)

def I3DResNet50(num_classes, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D(num_classes, AP3D.I3D, c3d_idx, nl_idx, **kwargs)

def API3DResNet50(num_classes, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D(num_classes, AP3D.API3D, c3d_idx, nl_idx, **kwargs)

def AP3DNLResNet50(num_classes, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[1, 3],[1, 3, 5],[]]

    return ResNet503D(num_classes, AP3D.APP3DC, c3d_idx, nl_idx, **kwargs)


model_dict = {
    # 'resnet18': [resnet18, 512],
    'c2dresnet50': [C2DResNet50, 2048],
    'medicalnet': [resnet50, 2048],
    'resnest50_3D': [resnest50_3D, 2048]
}

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='c2dresnet50', head='mlp', feat_dim=128, n_classes=2):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.sigma1 = nn.Parameter(torch.ones(1))
        self.sigma2 = nn.Parameter(torch.ones(1))
        self.encoder = model_fun(num_classes=n_classes)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat, x = self.encoder(x)
        
        feat = F.normalize(self.head(feat), dim=1)
        return feat,x 



if __name__ == '__main__':
    net = AP3DResNet50(5)
    x = torch.zeros(2,3,2,512,512)
    y,f = net(x)
    print(y.size(),f.size())