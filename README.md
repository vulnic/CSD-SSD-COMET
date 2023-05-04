# Consistency Optimization of ModEl-agnostic Transformations (COMET) 

By Vladimir Leung and Anthony Mangio

Originally by [Jisoo Jeong](http://mipal.snu.ac.kr/index.php/Jisoo_Jeong), Seungeui Lee, [Jee-soo Kim](http://mipal.snu.ac.kr/index.php/Jee-soo_Kim), [Nojun Kwak](http://mipal.snu.ac.kr/index.php/Nojun_Kwak): [link to their paper](https://papers.nips.cc/paper/9259-consistency-based-semi-supervised-learning-for-object-detection) 

## Installation & Preparation
Our project is tested on pytorch=1.10.0 and cuda=11.3.

#### prerequisites
- Python 3.7.1
- Pytorch 1.10.1
- torchvision 0.11.0
- matplotlib
- jupyterlab
- seaborn
- numpy
- pandas
- opencv-python
- pytorch_warmup


To run our project, run the following docker commands and pip install the requirements of this repository. We also recommend downloading the data outside of the docker container and mounting it:
```Shell
docker pull pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
docker run -it \
           --rm \
           --gpus all \
           -v <path_to_repository>:/code \
           pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime \
           /bin/bash
cd /code
pip install -r requirements.txt
```

## Download data
Once you have prepared the environment, you will need to download the data for this project. Use the following script to do so:
```Shell
cd /code
bash data/scripts/VOC2007.sh
bash data/scripts/VOC2012.sh
```
The default download location is `~/data` which we will use to refer to the download location. However, we recommend you move the downloaded data out of the container.

## SSD training
To train each of the supervised models, please run one of the two bash files below. If you run into trouble with `~/data`, please use the absolute path to this location:
 - `train_VOC07.sh ~/data`
 - `train_VOC0712.sh ~/data`

## COMET + SSD training
To train each the semi-supervised model, please run the bash file below. If you run into trouble with `~/data`, please use the absolute path to this location:
 - `train_VOC_COMET.sh ~/data`

## Evaluation
To evaluate a folder of models, please run the bash file below. If you run into trouble with `~/data`, please use the absolute path to this location:

```Shell 
python eval.py --all_weights <path_to_weights> \
               --voc_root ~/data
```
