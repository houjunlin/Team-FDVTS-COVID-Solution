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
parser.add_argument('--batch-size', '-bs', default=32, type=int, help='batch-size')
parser.add_argument('--n_classes', '-n_cls', default=2, type=int, help='n_classes')


def parse_args():
    global args
    args = parser.parse_args()


    return results


@torch.no_grad()
def test_all():
    parse_args()

    weight_dir = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/checkpoints/con/iccvsupcon128/74.pkl'

    result_file = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/results/val.csv'
    print(result_file)

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
    test_data = Lung3D_ccii_patient_supcon(train=False,val=True,inference=False,n_classes=args.n_classes)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    results = []

    correct = .0
    total_sample = .0
    con_matx = meter.ConfusionMeter(args.n_classes)
    prob_list = []
    label_list = []
    pred_list = []

    pbar = tqdm(test_loader, ascii=True)
    for i, (data,label,id) in enumerate(pbar):
        count = .0
        # non_ncp_list = []
        # ncp_list = []

        non_ncp_list = []
        ncp_list = []
        # cp_list = []


        cur_data = data.unsqueeze(1)
        cur_data = cur_data.float().cuda()
        label = label.float().cuda()

        _, pred = net(cur_data)
        pred = F.softmax(pred)
        _, predicted = pred.max(1)

        # print(pred[0])
        p_non_ncp = pred[0][0]
        p_ncp = pred[0][1]
        # p_cp = pred[0][2]

        total_sample += cur_data.size(0)
        correct += predicted.eq(label.long()).sum().item()        
        con_matx.add(predicted.detach(),label.detach())

        non_ncp_list.append(p_non_ncp.cpu().detach().item())
        ncp_list.append(p_ncp.cpu().detach().item())
        # cp_list.append(p_cp.cpu().detach().item())

        # prob_list.append(pred.cpu().detach())
        # label_list.append(label.cpu().detach())
        # pred_list.append(predicted.cpu().detach())

        prob_list.extend(pred.cpu().detach().tolist())
        pred_list.extend(predicted.cpu().detach().tolist())
        label_list.extend(label.cpu().detach().tolist())
        # print(np.array(prob_list).shape)

        # print(pred_list)
        # print(prob_list)
        # print(label_list)

        label = label.detach().tolist()
        batch_results = [(id_,label_,pred_.item(),non_ncp_,ncp_) for id_,label_,pred_,non_ncp_,ncp_ in zip(id,label,predicted.cpu().detach(),non_ncp_list,ncp_list) ]

        results += batch_results

        pbar.set_description(' acc: %.3f'% (100.* correct / total_sample))
        


    # recall = recall_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(), average=None)
    # precision = precision_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
    recall = recall_score(np.array(label_list), np.array(pred_list), average=None)
    precision = precision_score(np.array(label_list), np.array(pred_list),average=None)   
    print('recall:',recall*100.)
    print('precision:',precision*100.)

    # f1 = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(), average=None)
    # f1_macro = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(), average='macro')
    # f1 = f1_score(np.array(label_list), np.array(pred_list), average=None)
    f1_macro = f1_score(np.array(label_list), np.array(pred_list), average='macro')

    # specificity 
    # print('sen:',recall*100.)
    # for i in range(args.n_classes):
    #     TN = con_matx.value().sum() - con_matx.value()[i,:].sum() - con_matx.value()[:,i].sum() + con_matx.value()[i,i]
    #     FP = con_matx.value()[:,i].sum() - con_matx.value()[i,i]
    #     specificity = TN/(TN+FP)
    #     print('spec',i,':',specificity*100.)


    auc_score = roc_auc_score(np.array(label_list), np.array(prob_list)[:,1])
    print("auc = ", auc_score)

    write_csv(results, result_file)
    print(correct, total_sample)
    acc = 100.* correct/total_sample
    print('acc: ',acc)
    print('test cm:',str(con_matx.value()))

    # print('f1:',f1*100.)
    print('f1_macro:',f1_macro*100.)
    return results


def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        # writer.writerow(['id','label','health_rate','pcp_max_prob','pos_max_prob','pcp_noisyor','pos_noisyor','health_noisyor'])
        writer.writerow(['id','label','pred','non-covid','covid'])
        writer.writerows(results)


if __name__ == '__main__':
    test_all()
        
