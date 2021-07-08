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




def write_csv(results,file_name):
    import csv
    csvfile = open(file_name,"w",newline = "")   
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

    seed_everything(CFG['seed'])

    test_dataset = Lung3D_ccii_patient_supcon(train=False,val=False,inference=True,n_classes=CFG['num_classes'])
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = SupConResNet(name='c2dresnet50', head='mlp', feat_dim=128, n_classes=CFG['num_classes'])
    model = nn.DataParallel(model)
    model.cuda()

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

    print("noncovid number: ", noncovid_count, " covid number: ", covid_count)

    write_csv(noncovid_list,non_covid_result_file)
    write_csv(covid_list,covid_result_file)

