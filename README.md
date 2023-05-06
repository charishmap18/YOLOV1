# YOLOV1
### Team Members -
Charishma Paruchuri (researched about YOLOV1, implementde yolo, implemented loss functions and batch normalization)

Deepika Narne (researched and implemented YOLO V1, trained and tested the model and implemented batch normalization)

#### Instructions to run the model:
1. GPU is preferred for faster implementation. (Through Google Colab, free GPU may be used by switching the runtime type to GPU.)
2. Need to upload dataset.py and utils.py as external libraries to download a dataset from Kaggle into Google Colab. Dataset used for the project: Pascal voc (Size: 5GB)
3. Also need to upload kaggle.json to access kaggle datasets ( Find instructions to download kaggle.json below).
4. Two implementations with and without batch normalization are uploaded seperately to understand clear outcome differences.

#### Here are the steps to download the Kaggle API key file, also known as the kaggle.json file:
1. Log in to your Kaggle account on the Kaggle website (https://www.kaggle.com/).
2. Click on your profile picture in the top-right corner of the screen and select "Account" from the dropdown menu.
3. Scroll down to the "API" section, and click on the "Create New API Token" button. This will download the kaggle.json file to your local machine.

## Implementation details:

We re-implemented the YOLO(v1) Architecture along with applying Batch Normalization using Pascal VOC data set. The YOLO (You Only Look Once) algorithm is a popular object detection algorithm that uses a single convolutional neural network (CNN) to simultaneously predict object bounding boxes and class probabilities. In the YOLO(v1) architecture, the input images are resized to a fixed size of 448 x 448 pixels. This is done to ensure that the input size is consistent across all images and to simplify the architecture of the network.

The YOLO network consists of 24 convolutional layers followed by 2 fully connected layers. The convolutional layers are used to extract features from the input image, while the fully connected layers are used to make predictions about the object classes and bounding box coordinates. This is done by dividing the image into a grid of cells and predicting a set of bounding boxes and class probabilities for each cell. The bounding box coordinates are represented as offsets from the center of each cell, which helps to improve the accuracy of the bounding box predictions.

The YOLO(v1) architecture was trained using the VOC (Visual Object Classes) dataset, which contains images of 20 different object classes, such as cars, pedestrians, and bicycles. The training process involves optimizing the weights of the YOLO network, with the goal of minimizing the loss function, which is a combination of the classification and localization errors.

# Output Graphs of loss functions

### YOLO V1 before implementing Batch Normalization.)

![download1](https://user-images.githubusercontent.com/132419470/236590142-7d92b7eb-8259-4997-bc31-82cce1ffc34a.png)

### Loss functions graph for YOLO V1 with batch Normalization Included

![download2](https://user-images.githubusercontent.com/132419470/236590209-c2fdcb58-a1bb-4f74-b2a4-be4e99962338.png)

# References:

[1] Original Paper: https://arxiv.org/abs/1506.02640v5

Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[2] Original Paper Implementation: https://github.com/JeffersonQin/yolo-v1-pytorch , https://pjreddie.com/darknet/yolo/

[3] Dataset: https://www.kaggle.com/datasets/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video.
