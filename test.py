#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import argparse
import os
import torch.nn as nn

from torch.utils.data import DataLoader
from datetime import datetime
from functions import progress_bar
from visualize import Visualizer
from torchnet import meter
import torch.nn.functional as F
from sklearn.metrics import recall_score,precision_score,f1_score,roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from models.ResNet import SupConResNet

from dataset1 import Lung3D_ccii_patient_supcon
from tqdm import tqdm


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default=2, type=int,help='model')
parser.add_argument('--pre', '-p', default=0, type=int,help='pre1-nopre0')
parser.add_argument('--batch-size', '-bs', default=4, type=int, help='batch-size')
parser.add_argument('--n_classes', '-n_cls', default=2, type=int, help='n_classes')


def parse_args():
    global args
    args = parser.parse_args()



@torch.no_grad()
def test_all():
    parse_args()

    weight_dir = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/checkpoints/con/iccvimgsupcon128/53.pkl'

    covid_result_file = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/test/con/iccvimgsupcon128/covid.csv'
    non_covid_result_file = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/test/con/iccvimgsupcon128/non-covid.csv'

    ckpt = torch.load(weight_dir)
    state_dict = ckpt['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name =k[7:]
        new_state_dict[name] = v

    net = SupConResNet(name='c2dresnet50', head='mlp', feat_dim=128, n_classes=args.n_classes)
    net.load_state_dict(new_state_dict)

    net = nn.DataParallel(net)
    net = net.cuda()
    net = net.eval()

    # data
    test_data = Lung3D_ccii_patient_supcon(train=False,val=False,inference=True,n_classes=args.n_classes)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    noncovid_count = .0
    covid_count = .0
    noncovid_list = []
    covid_list = []

    pbar = tqdm(test_loader, ascii=True)
    for i, (data,_,id) in enumerate(pbar):
        cur_data = data.unsqueeze(1)
        cur_data = cur_data.float().cuda()

        _, pred = net(cur_data)
        pred = F.softmax(pred)
        _, predicted = pred.max(1)

        name = id[0].split('/')[-1].split('.')[0]
        print(name)

        if predicted == 0:
            noncovid_list.append(name)
            noncovid_count += 1
        if predicted == 1:
            covid_list.append(name)
            covid_count += 1
    
    print("noncovid number: ", noncovid_count, " covid number: ", covid_count)

    write_csv(noncovid_list,non_covid_result_file)
    write_csv(covid_list,covid_result_file)

 

def write_csv(results,file_name):
    import csv
    csvfile = open(file_name,"w",newline = "")  
    writer = csv.writer(csvfile)
    writer.writerow(results)
    csvfile.close()


if __name__ == '__main__':
    test_all()
        
