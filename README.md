# Team-FDVTS-COVID-Solution


This is the official implementation of the paper "CMC-COV19D: Contrastive Mixup Classification for COVID-19 Diagnosis". 
We ranked 1st in the ICCV 2021 COVID-19 Diagnosis Competition.

## Setup
+ Pillow==8.1.2
+ scikit-image==0.18.2
+ scikit-learn==0.24.2
+ scipy==1.6.1
+ torch==1.8.0
+ torchfile==0.1.0
+ torchnet==0.0.4
+ torchvision==0.9.0
+ visdom==0.1.8.9

## Data Preprocessing


## Train
```
python main_supcon.py
```

## Test
```
python test.py
```

## TTA Test
```
python tta_val.py
```


## Reference
```
@InProceedings{Hou_2021_ICCV,
    author    = {Hou, Junlin and Xu, Jilan and Feng, Rui and Zhang, Yuejie and Shan, Fei and Shi, Weiya},
    title     = {CMC-COV19D: Contrastive Mixup Classification for COVID-19 Diagnosis},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2021},
    pages     = {454-461}
}
```
