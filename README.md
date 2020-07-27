# PHNet: Parasite-Host Network for Video Crowd Counting
![Figure](https://github.com/LeeJAJA/PHNet-pytorch/blob/master/result.jpg)

## Training

### Pretreatment

1. Download [Venice Dataset](https://sites.google.com/view/weizheliu/home/projects/context-aware-crowd-counting) as example, unzip and move Venice dataset into `./PHNet-pytorch/dataset/Venice`
2. Perform pretreatments in order(density map, ROI and 3D dataset generation)

### Train

#### From initial

`python model/train.py /dataset/Venice/train_data.json /dataset/Venice/test_data.json 1 --gpu 0,1 -bs 8`

#### From pretrained model

`cd model && wget https://github.com/LeeJAJA/PHNet-pytorch/releases/download/1.0/PHnet_checkpoint.pth`

`python model/train.py /dataset/Venice/train_data.json /dataset/Venice/test_data.json 1 --gpu 0,1 -bs 8 --pre /model/PHnet_checkpoint.pth`

## Authors & Contributors

PHNet is authored by [Jiajie Li](https://github.com/LeeJAJA), Shiqiao Meng, Weiwei Guo, Lai Ye and Jinfeng Jiang from Tongji University. [Jiajie Li](https://github.com/LeeJAJA) is the corresponding author. The code is developed by [Jiajie Li](https://github.com/LeeJAJA) and Shiqiao Meng. Currently, it is being maintained by [Jiajie Li](https://github.com/LeeJAJA).

