# Playing-Cards-Detection
This will give you a tutorial on how to generate and train a model with Yolov8 and deploy it on Jetson Device.

# Training Reference by Yolov8

##### Google Colab : https://colab.research.google.com/drive/1muD6Xzu11rw_en0kkPIM9-aZtlXaaSKT?usp=sharing
### Steps in this Tutorial

In this tutorial:

- Before you start
- Install YOLOv8
- Custom Training
- Validate Custom Model
- Inference with Custom Model
- Deploy it on Jetson Device
**Let's begin!**

## Installation

Let's make sure that we have access to GPU. We can use `nvidia-smi` command to do that. In case of any problems navigate to `Edit` -> `Notebook settings` -> `Hardware accelerator`, set it to `GPU`, and then click `Save`.

## Next we import os so that we can create or remove the file.
```python
import os
HOME = os.getcwd()
print(HOME)
```
## Install YOLOv8

We install by using !pip install ultralytics and check if it is installed or not.
```python
# Pip install method (recommended)

!pip install ultralytics==8.0.20

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()
```
```python
from ultralytics import YOLO

from IPython.display import display, Image
```


## Roboflow Universe

We already create the dataset in the Roboflow so we will skip this part. or you can see this document: "https://docs.roboflow.com/adding-data"

#### We make a directory name 'HOME' and use function !pip install roboflow so that we can import our labelized datasets.
```python
!mkdir {HOME}/datasets
%cd {HOME}/datasets

!pip install roboflow --quiet

from roboflow import Roboflow
rf = Roboflow(api_key="iu2x3rAOfACxzeU1N4qp")
project = rf.workspace("kmitl-kln1o").project("playingcardsdetection-new-4uooq")
dataset = project.version(2).download("yolov8")
```


## Custom Training

1. Use model yolov8s (fast and high accuracy)
2. Located the datasets location.
3. Epoch = 80
4. Image size = 640
```python
%cd {HOME}

!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=80 imgsz=640 plots=True
```

## List if the training create our file
```python
!ls {HOME}/runs/detect/train/
```
## Show the confusion Matrix and graph to view the performance of our training model.
```python
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)
```
## Show to performance
```python
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)
```

## Test our model on examples
```python
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=600)
```

## Validate Custom Model
```python
%cd {HOME}

!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
```

## Inference with Custom Model
```python
%cd {HOME}
!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True
```
## Deploy the model on Roboflow
```python
project.version(dataset.version).deploy(model_type="yolov8", model_path=f"{HOME}/runs/detect/train/")
```
```python
#Run inference on your model on a persistant, auto-scaling, cloud API

#load model
model = project.version(dataset.version).model

#choose random test set image
import os, random
test_set_loc = dataset.location + "/test/images/"
random_test_image = random.choice(os.listdir(test_set_loc))
print("running inference on " + random_test_image)

pred = model.predict(test_set_loc + random_test_image, confidence=40, overlap=30).json()
pred
```

# Deploy it on Jetson
Don't forget to upload best.pt to the Jetson device.
## Install the required package
```bash
!sudo apt update
!sudo apt install -y python3.8 python3.8-venv python3.8-dev python3-pip \
!libopenmpi-dev libomp-dev libopenblas-dev libblas-dev libeigen3-dev libcublas-dev
```

## Clone the YOLOv8 repository.
```bash
!python3.8 -m venv venv
!source venv/bin/activate
```

## Create a Python 3.8 virtual environment using venv.
```bash
!python3.8 -m venv venv
!source venv/bin/activate
```

## Update Python packages not specified in YOLOv8.
```bash
!pip install -U pip wheel gdown
```

## Download and install the pre-built PyTorch, TorchVision package. This package was built using the method described in this article. This article also uses the pre-built package.
```bash
# pytorch 1.11.0
!gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM
# torchvision 0.12.0
!gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV
!python3.8 -m pip install torch-*.whl torchvision-*.whl
```

## Install the Python package for YOLOv8. (This will run setup.py)
```bash
!pip install .
```

## Execute object detection.
```bash
!yolo task=detect mode=predict model=best.pt source=0 show=True
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
