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
from dataset1 import Lung3D_ccii_patient_supcon, Lung3D_ccii_patient_clf
from torch.utils.data import DataLoader
import torch.nn.functional as F
from visualize import Visualizer
from torchnet import meter
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from models.ResNet import C2DResNet50
from models.resnet3D import resnet34,resnet50
import random
import torch.backends.cudnn as cudnn
import logging

print("torch = {}".format(torch.__version__))

IMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--visname', '-vis', default='2d', help='visname')
parser.add_argument('--batch_size', '-bs', default=16, type=int, help='batch_size')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n_cls', default=2, type=int, help='n_classes')
parser.add_argument('--distpre', '-pre', default=False, type=bool, help='use pretrained')

best_f1 = 0
val_epoch = 5
save_epoch = 5

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

def main():
    global best_f1
    global save_dir

    parse_args()
    vis = Visualizer(args.visname,port=9000)

    if not args.distpre:
        target_model = resnet50(num_classes=args.n_classes) 

        # target_model = C2DResNet50(args.n_classes)
        print("MODEL: resnet50 imagenet!!!")
    else:
        target_model = resnet50(pretrained=False) 
        weight_dir = '/remote-home/my/2019-nCov/CC-CCII/distmap/checkpoints/cciidistpre/90.pkl'
        print("load pre weight...",weight_dir.split('/')[-1])
        checkpoint = torch.load(weight_dir)
        state_dict = checkpoint['net']

        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.features.", "")] = state_dict[key]
            # unParalled_state_dict[key] = state_dict[key]

        target_model.load_state_dict(unParalled_state_dict, strict=False) 
        target_model.fc = nn.Linear(2048,args.n_classes) 

    # medicalnet
    '''
    weight_dir = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/medicalnet/resnet_50_23dataset.pth'
    print("load pre weight...",weight_dir.split('/')[-1])
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    target_model.load_state_dict(unParalled_state_dict, strict=False) 
    '''

    target_model = nn.DataParallel(target_model)
    target_model = target_model.cuda()

    train_data = Lung3D_ccii_patient_clf(train=True,val=False,inference=False,n_classes=args.n_classes)
    val_data = Lung3D_ccii_patient_clf(train=False,val=True,inference=False,n_classes=args.n_classes)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(target_model.parameters(), args.lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
    optimizer = torch.optim.Adam(target_model.parameters(), args.lr, weight_decay=1e-5)

    con_matx = meter.ConfusionMeter(args.n_classes)

    save_dir = './checkpoints/'+ str(args.visname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  

    # log
    # if not os.path.exists(snapshot_path):
    #     os.makedirs(snapshot_path)
    #     os.makedirs(snapshot_path + './checkpoint')
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=save_dir+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # train the model

    # initial_epoch = checkpoint['epoch']
    initial_epoch = 0
    for epoch in range(initial_epoch, initial_epoch + args.epochs):
        target_model.train()
        con_matx.reset()
        total_loss = .0
        total = .0
        correct = .0
        count = .0
        total_num = .0

        lr = get_lr(epoch, args.epochs)
        # lr = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        pred_list = []
        label_list = []
        
        pbar = tqdm(train_loader, ascii=True)

        for i, (imgs, labels, id) in enumerate(pbar):
            # augment = 1
            # print(i,imgs.shape,labels,id)
            imgs = imgs.unsqueeze(1).float().cuda()
            labels = labels.float().cuda()
            pred = target_model(imgs)
            pred = F.softmax(pred)

            con_matx.add(pred.detach(),labels.detach())
            _, predicted = pred.max(1)
            loss = criterion(pred, labels.long())
            
            pred_list.append(predicted.cpu().detach())
            label_list.append(labels.cpu().detach())

            total_loss += loss.item()
            total += labels.shape[0]
            correct += predicted.eq(labels.long()).sum().item()

            # augment = 2
            # imgs = torch.cat([imgs[0],imgs[1]],dim=0) #2*bsz,64,256,256
            # # imgs = imgs.unsqueeze(1).float().cuda()
            # labels = labels.float().cuda()
            # bsz = labels.shape[0]

            # _, pred = target_model(imgs) #2*bsz,128 #2*bsz,4
            # pred1, pred2 = torch.split(pred, [bsz, bsz], dim=0) #bsz,4
            # pred1 = F.softmax(pred1)
            # pred2 = F.softmax(pred2)
            # con_matx.add(pred1.detach(),labels.detach())
            # con_matx.add(pred2.detach(),labels.detach())
            # _, predicted1 = pred1.max(1)
            # _, predicted2 = pred2.max(1)
            # loss = 0.5*criterion(pred1,labels.long())+0.5*criterion(pred2,labels.long())

            # pred_list.append(predicted1.cpu().detach())
            # label_list.append(labels.cpu().detach())
            # pred_list.append(predicted2.cpu().detach())
            # label_list.append(labels.cpu().detach())

            # total_loss += loss.item()
            # total += 2*labels.shape[0]

            # correct += predicted1.eq(labels.long()).sum().item()
            # correct += predicted2.eq(labels.long()).sum().item()
            
            count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    
            pbar.set_description('loss:%.3f' % (total_loss / (i+1)) + ' acc: %.3f'% (100*correct/total))
                    
            #progress_bar(i, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    #   % (total_loss / (i + 1), 100. * correct / total, correct, total))
        recall = recall_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
        precision = precision_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
        vis.plot('loss', total_loss/count)
        vis.log('epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},recall:{recall},precision:{precision}'.format(epoch=epoch,lr=lr,loss=total_loss/count,train_cm=str(con_matx.value()),recall=recall,precision=precision))
        
        logging.info('epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},recall:{recall},precision:{precision}'.format(epoch=epoch,lr=lr,loss=total_loss/count,train_cm=str(con_matx.value()),recall=recall,precision=precision))

        if (epoch + 1) % val_epoch == 0:
            val(target_model,val_loader,epoch,vis, logging)

@torch.no_grad()
def val(net, val_loader, epoch, vis, logging):
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
        data = data.unsqueeze(1).float().cuda()
        label = label.float().cuda()
        pred = net(data)
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

    logging.info(' VAL epoch:{epoch},val_acc:{val_acc},val_cm:{val_cm},recall:{recall},precision:{precision},f1:{f1}'.format(epoch=epoch,val_acc=acc,val_cm=str(con_matx.value()),recall=recall,precision=precision,f1=f1))
    logging.info('\n')

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
