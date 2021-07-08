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
# from draw_roc import draw_roc
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



'''all: h1n1 / h1h1 cbam (no mask)'''
@torch.no_grad()
def test_all():
    parse_args()

    weight_dir = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/checkpoints/con/iccvsupcon128/74.pkl'
    # print("4res18nopresupcon3cv1")

    result_file = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/results/val.csv'
    print(result_file)
    # png_name = 'roc/ccii3d2clfres50supcon'

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

    # draw_roc(np.array(label_list), np.array(prob_list),png_name,n_classes=args.n_classes)
    # draw_roc(torch.cat(label_list).numpy(), torch.cat(prob_list).numpy(),png_name,n_classes=args.n_classes)
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
    csvfile = open(file_name,"w",newline = "")      #w是覆盖形写入，a是追加写入
    writer = csv.writer(csvfile)
    writer.writerow(results)
    csvfile.close()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

CFG = {
    'seed': 42,
    'img_size': 960,
    'valid_bs': 20,
    'num_workers': 4,
    'num_classes': 2,
    'tta': 3,
    'models':['/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/checkpoints/con/iccvsupcon128/74.pkl',
              '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/checkpoints/con/iccvimgsupcon128/53.pkl'
            ],
    'base_img_path': '/remote-home/share/18-houjunlin-18110240004/病灶点标注任务/data',
    'weights': [1,1]
}



def seed_everything(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def validate(model, loader): 
    model.eval()
    preds = []
    labels = []
    names = []
    pbar = tqdm(enumerate(loader), total=len(loader))  
    with torch.no_grad():
        for batch_idx, (input, label ,id) in pbar:
            input = input.unsqueeze(1).float().cuda()
            # label = label.float().cuda()
            _, output = model(input)
            preds.append(F.softmax(output).to('cpu').numpy())
            # labels.append(label.to('cpu').numpy())

            name = id[0].split('/')[-1].split('.')[0]
            names.append(name)
    predictions = np.concatenate(preds)
    # labels = np.concatenate(labels)
    # names = np.concatenate(names)
    labels = names
    return predictions, labels, names

if __name__ == '__main__':
  
    covid_result_file = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/test/con/tta/covid.csv'
    non_covid_result_file = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/test/con/tta/non-covid.csv'


    # from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
    # import pandas as pd
    # import torch.utils.data as data
    seed_everything(CFG['seed'])

    test_dataset = Lung3D_ccii_patient_supcon(train=False,val=False,inference=True,n_classes=CFG['num_classes'])
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = SupConResNet(name='c2dresnet50', head='mlp', feat_dim=128, n_classes=CFG['num_classes'])
    model = nn.DataParallel(model)
    model.cuda()


    # data_ = pd.read_csv('/remote-home/share/18-houjunlin-18110240004/多标签病灶点分类/multi_label2823.csv')
    # test_index = [i for i in range(data_.shape[0])]

    # test_data = data_.iloc[test_index, :].reset_index(drop=True) 
    # test_transforms = get_riadd_test_transforms(CFG)
    # test_dataset = RiaddDataSet9Classes(image_ids = test_data,transform = test_transforms, 
    #                             baseImgPath = CFG['base_img_path'])
    # test_data_loader = data.DataLoader( test_dataset, 
    #                                     batch_size=CFG['valid_bs'], 
    #                                     shuffle=False, 
    #                                     num_workers=CFG['num_workers'], 
    #                                     pin_memory=True, 
    #                                     drop_last=False,
    #                                     sampler = None)
    
    # imgIds = test_data.iloc[test_index,0].tolist()
    # target_cols = test_data.iloc[test_index, 1:].columns.tolist()    
    # test = pd.DataFrame()
    # test['ID'] = imgIds

    tst_preds = []
    for i,model_name in enumerate(CFG['models']):
        model_path = os.path.join(model_name)

        print('model_path: ',model_path)

        ckpt = torch.load(model_path)
        state_dict = ckpt['net']
        model.load_state_dict(state_dict, strict=True)

        for _ in range(CFG['tta']):
            preds, labels, names = validate(model,test_data_loader)
            tst_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*preds]
    tst_preds = np.sum(tst_preds, axis=0)
    # print(labels.shape, tst_preds.shape)
    # print(labels)
    print(tst_preds)

    tst_predicted = []

    noncovid_list = []
    covid_list = []
    noncovid_count=0
    covid_count=0
    for i in range(tst_preds.shape[0]):
        if tst_preds[i][0] > tst_preds[i][1]:
            tst_predicted.append(0)
            noncovid_list.append(names[i])
            noncovid_count += 1
        else:
            tst_predicted.append(1)
            covid_list.append(names[i])
            covid_count += 1
    # tst_predicted = tst_preds.max(1)
    # print(tst_predicted)
    # f1_macro = f1_score(np.array(labels), np.array(tst_predicted), average='macro')
    # print(f1_macro)

    print("noncovid number: ", noncovid_count, " covid number: ", covid_count)

    write_csv(noncovid_list,non_covid_result_file)
    write_csv(covid_list,covid_result_file)


    # test[target_cols] = tst_preds
    # test.to_csv('submission3.csv', index=False)