# Updated Version of YOLOv5 with CUDA Fix

## Description

- Integrated from yolov5 official repo, fixed the issue on cuda core can't run on yolov7 model. Tested on conda environment with `python3.9, pytorch=1.11.0, cudatoolkit=1.13`

## Installation on Windows 

Conda environment -- Anaconda https://www.anaconda.com/ \
Python -- Python 3.9 installed with Anaconda

### Install Anaconda
Select Installation Type : 'just me' \
Anaconda Install Location : Anywhere you want, doesn't have to be on C drive \
Advanved Installation Options : \
![image](https://user-images.githubusercontent.com/56321690/202285992-e6f95310-7aa7-4997-a186-059bd7886b8d.png)


### Clone yolov5 Repo 
Run `git clone https://github.com/FlyerJB/YOLOv7-RoboMaster.git` on your command prompt to some dir under C:/ drive or your OS drive to avoid Enviornment failure \
    
### Make Conda Environment
Open Conda Command Prompt with Admin Right \
Cd into yolov5 dir with `cd <where you clone your yolov5>` \
And craete Conda Environment
`Conda Create -n <The Name You Like> Python3.9` 

![image](https://user-images.githubusercontent.com/56321690/202287922-1a6b9a71-49ef-4d40-b759-ec4ddd641317.png)
### Activate Conda Environment
Run command `Conda activate <The Name You put from previous step>` 

### pip install required packages
Option 1 : Install yolov5 for training on CPU \
`pip install -r requirements.txt`

Option 2 : Install yolov5 for training on RTX GPU \
`pip install -r requirement_nv_gpu.txt` \
`pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`

### Validate Cuda Installation ( required for nv_gpu training )
Run `python` or `python3` or `py` to run python \
Run `import torch` \
Run `torch.cuda.is_available()` \
If it returns `True`, it means CUDA is successfully install on your device with Pytorch. \
Then run `exit()` to exit python

## Seperate Dataset
### Validate set
10% - 20% of the whole dataset \
It goes under `data\val` foler with train and label folders inside \
![image](https://user-images.githubusercontent.com/56321690/203417152-20db03aa-b29c-4b08-8320-9cd3b3df118b.png)

### Train set
80% to 90% of the whole dataset \
It goes under `data\train` foler with train and label folders inside \
![image](https://user-images.githubusercontent.com/56321690/203417152-20db03aa-b29c-4b08-8320-9cd3b3df118b.png)


## Trained model
Our own model have been up uploaded to `cv-yolov5/model` files
[**Models**](https://github.com/macrobomaster/cv-yolov5/tree/main/models)

## Training

With GPU training

``` shell
# train models
python train.py --workers 1 --device 0 --batch-size -1 --epochs 50 --img 640 --data data/coco_custom.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov5-custom.yaml --name yolov5-tut3 --weights yolov5.pt  
```
With CPU training

``` shell
# train models
python train.py --workers 8 --device CPU --batch-size -1 --epochs 50 --img 640 --data data/coco_custom.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov5-custom.yaml --name yolov5-tut3 --weights yolov5.pt
```

