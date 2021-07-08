#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import sys
from utils import *
from tqdm import tqdm
from dataset1 import Lung3D_ccii_patient_supcon
from torch.utils.data import DataLoader
import torch.nn.functional as F
from visualize import Visualizer
from torchnet import meter
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from models.ResNet import SupConResNet

import torch.backends.cudnn as cudnn
import random
import math


print("torch = {}".format(torch.__version__))

IMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--visname', '-vis', default='2d', help='visname')
parser.add_argument('--batch_size', '-bs', default=4, type=int, help='batch_size')
parser.add_argument('--lr', '-lr', default=1e-4, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n_cls', default=2, type=int, help='n_classes')
parser.add_argument('--distpre', '-pre', default=False, type=bool, help='use pretrained')


best_f1 = 0
val_epoch = 1
save_epoch = 10

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.deterministic = True


def parse_args():
    global args
    args = parser.parse_args()

def get_lr(cur, epochs):
    if cur < int(epochs * 0.3):
        lr = args.lr
    elif cur < int(epochs * 0.8):
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    return lr

def get_dynamic_lr(cur, epochs):
    power = 0.9
    lr = args.lr * (1 - cur / epochs) ** power
    return lr


def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a.long(), y_b.long(), lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    print(torch.cuda.device_count())
    global best_f1
    global save_dir

    parse_args()
    vis = Visualizer(args.visname,port=9000)

    # prepare the model
    target_model = SupConResNet(name='resnest50_3D', head='mlp', feat_dim=128, n_classes=args.n_classes)

    s1 = target_model.sigma1
    s2 = target_model.sigma2

    # ccii-pre
    # ckpt = torch.load('/remote-home/junlinHou/2019-nCov/CC-CCII/3dlung/checkpoints/con/ccii3d2clfres50supcon/19.pkl')
    # state_dict = ckpt['net']
    # unParalled_state_dict = {}
    # for key in state_dict.keys():
    #     unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    # target_model.load_state_dict(unParalled_state_dict,True)

    target_model = nn.DataParallel(target_model)
    target_model = target_model.cuda()
    
    # prepare data
    train_data = Lung3D_ccii_patient_supcon(train=True,val=False,n_classes=args.n_classes)
    val_data = Lung3D_ccii_patient_supcon(train=False,val=True,n_classes=args.n_classes)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4,pin_memory=True)

    criterion = SupConLoss(temperature=0.1)
    criterion = criterion.cuda()
    criterion_clf = nn.CrossEntropyLoss()
    criterion_clf = criterion_clf.cuda()
    optimizer = torch.optim.Adam(target_model.parameters(), args.lr, weight_decay=1e-5)

    con_matx = meter.ConfusionMeter(args.n_classes)

    save_dir = './checkpoints/con/'+ str(args.visname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  

    # train the model

    # initial_epoch = checkpoint['epoch']
    initial_epoch = 0
    for epoch in range(initial_epoch, initial_epoch + args.epochs):

        target_model.train()
        con_matx.reset()
        total_loss1 = .0
        total_loss2 = .0
        total = .0
        correct = .0
        count = .0
        total_num = .0

        lr = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        pred_list = []
        label_list = []
        
        pbar = tqdm(train_loader, ascii=True)
   
        for i, (imgs, labels, ID) in enumerate(pbar):
                
            imgs = torch.cat([imgs[0],imgs[1]],dim=0) #2*bsz,256,256
            imgs = imgs.unsqueeze(1).float().cuda() #2*bsz,1,256,256
            labels = labels.float().cuda()
            bsz = labels.shape[0]


            ## mixup
            lam = 0.5
            imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, lam, use_cuda)
            features, pred = target_model(imgs)
            pred1, pred2 = torch.split(pred, [bsz, bsz], dim=0) #bsz,n_classs
            targets_a1, targets_a2 = torch.split(targets_a, [bsz, bsz], dim=0) #bsz,n_classs
            targets_b1, targets_b2 = torch.split(targets_b, [bsz, bsz], dim=0) #bsz,n_classs


            pred1 = F.softmax(pred1)
            pred2 = F.softmax(pred2)
            con_matx.add(pred1.detach(),labels.detach())
            con_matx.add(pred2.detach(),labels.detach())
            _, predicted1 = pred1.max(1)
            _, predicted2 = pred2.max(1)

            loss_func1 = mixup_criterion(targets_a1, targets_b1, lam)
            loss_mixup1 = loss_func1(criterion_clf, pred1)

            loss_func2 = mixup_criterion(targets_a2, targets_b2, lam)
            loss_mixup2 = loss_func1(criterion_clf, pred2)
            
            loss_clf = 0.5*loss_mixup1+0.5*loss_mixup2

            # features, pred = target_model(imgs) #2*bsz,128 #2*bsz,n_class
            # f1, f2 = torch.split(features, [bsz, bsz], dim=0) #bsz,128
            # features = torch.cat([f1.unsqueeze(1),f2.unsqueeze(1)],dim=1) #bsz,2,128
            # loss_con = criterion(features,labels)

            # pred1, pred2 = torch.split(pred, [bsz, bsz], dim=0) #bsz,n_classs
            # pred1 = F.softmax(pred1)
            # pred2 = F.softmax(pred2)
            # con_matx.add(pred1.detach(),labels.detach())
            # con_matx.add(pred2.detach(),labels.detach())
            # _, predicted1 = pred1.max(1)
            # _, predicted2 = pred2.max(1)
            # loss_clf = 0.5*criterion_clf(pred1,labels.long())+0.5*criterion_clf(pred2,labels.long())

            pred_list.append(predicted1.cpu().detach())
            label_list.append(labels.cpu().detach())
            pred_list.append(predicted2.cpu().detach())
            label_list.append(labels.cpu().detach())

            loss = torch.exp(-s1)*loss_con+s1+torch.exp(-s2)*loss_clf+s2
            # loss = loss_clf
            total_loss1 += loss_con.item()
            total_loss2 += loss_clf.item()
            total += 2 * bsz
            correct += predicted1.eq(labels.long()).sum().item()
            correct += predicted2.eq(labels.long()).sum().item()
            count += 1
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    
            pbar.set_description('loss: %.3f' % (total_loss2 / (i+1))+' acc: %.3f' % (correct/total))

        recall = recall_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
        precision = precision_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
        vis.plot('loss1_con', total_loss1/count)
        vis.plot('loss', total_loss2/count)
        vis.log('epoch:{epoch},lr:{lr},loss1:{loss1},loss2:{loss2}'.format(epoch=epoch,lr=lr,loss1=total_loss1/count,loss2=total_loss2/count))
        
        if (epoch + 1) % val_epoch == 0:
            val1(target_model,val_loader,epoch,vis)
            print(torch.exp(-s1).item(),torch.exp(-s2).item())
         

@torch.no_grad()
def val1(net, val_loader, epoch, vis):
    global best_f1
    parse_args()
    net = net.eval()

    correct = .0
    total = .0
    con_matx = meter.ConfusionMeter(args.n_classes)
    pred_list = []
    label_list = []
  
    pbar = tqdm(val_loader, ascii=True)
    for i, (data, label,id) in enumerate(pbar):
        data = data.unsqueeze(1)
        data = data.float().cuda()
        label = label.float().cuda()
        _, pred = net(data)
        pred = F.softmax(pred)
        _, predicted = pred.max(1)

        pred_list.append(predicted.cpu().detach())
        label_list.append(label.cpu().detach())

        total += data.size(0)
        correct += predicted.eq(label.long()).sum().item()        
        con_matx.add(predicted.detach(),label.detach()) 
        pbar.set_description(' acc: %.3f'% (100.* correct / total))


    recall = recall_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
    precision = precision_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
    f1 = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average='macro')
    
    print(correct, total)
    acc = 100.* correct/total

    print('val epoch:', epoch, ' val acc: ', acc, 'recall:', recall, "precision:", precision, "f1_macro:",f1)
    vis.plot('val acc', acc)
    vis.plot('val f1 macro', f1)
    vis.log('epoch:{epoch},val_acc:{val_acc},val_cm:{val_cm},recall:{recall},precision:{precision},f1:{f1}'.format(epoch=epoch,val_acc=acc,val_cm=str(con_matx.value()),recall=recall,precision=precision,f1=f1))   

    if f1 >= best_f1:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'f1': f1,
            'epoch': epoch,
        }
        save_name = os.path.join(save_dir, str(epoch) + '.pkl')
        torch.save(state, save_name)
        best_f1 = f1


if __name__ == "__main__":
    main()
        

