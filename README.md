# [ Based on CSD: Consistency-based Semi-supervised learning for object Detection (NeurIPS 2019)](https://papers.nips.cc/paper/9259-consistency-based-semi-supervised-learning-for-object-detection) 

## COMET SSD300
By [Vladimir Leung & Anthony Mangio)

## Installation & Preparation
This repository is dockerized and can be pulled down as a container. Datasets can be pulled down using scripts located CSD-SSD-COMET/data/scripts/. Network weights are saved in CSD-SSD-COMET/weights/. 

#### prerequisites
- Python 3.6
- Pytorch 1.13.1
- torchvision 0.14.1
- torchaudio 0.13.1
- matplotlib
- jupyterlab
- seaborn
- numpy
- pandas
- opencv-python
- pytorch_warmup

## Supervised learning for SSD300 on VOC2007 dataset
```Shell
python train_ssd.py
```

## COMET training
```Shell
python train_comet_ssd.py
```

## Evaluation Model Performance
```Shell
python eval.py
```
## SSD & COMET Model Locations

ssd.py
ssd_comet.py


