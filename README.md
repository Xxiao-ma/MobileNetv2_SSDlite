# Modified Mobilenet_v2_SSDlite with Background Subtraction Fusion Network
A keras version of real-time object detection network: mobilenet_v2_ssdlite

the pretrained weights file in the 'pretrained_weights' folder

the model structure in the 'model' folder

the train and inference process in the 'experiments' folder

## Dependencies:
* Python 3.x
* Numpy
* TensorFlow 1.3
* Keras 2.2.4
* OpenCV
* Beautiful Soup 4.x

# Baseline:
Modified MobileNetv2 + SSDlite:
notebook: mobileNetv2_ssdLite.ipynb





# Object tracker:

This is a Object Tracker based on Kalman Filter. Two extra rules are integrated into the tracker so that it can achieve higher counting accuracy in the use case of traffic counting.

1. Occlusion Tolerant Data Association
All generated trackers will keep updating and making predictions, until the prediction exceeds the boundary of the scenary. In this way, some unmatched tracker due to False Positive detection results or occlusion can be preserved and potentially be resued later. Thus, this can reduece the situation of counting a same object multiple times.

2. False Positive Sequence Filtering
Due to the limitation of the lightweight detection model, the model can sometimes generating False Positive results by mistaking background as target objects. By checking the inference results of the detection model, such False Positives are generated randomly and normally will not last for a long time. So the filter with a threshold of minimal lasting frames is integrated to filter the tracked objects. The filter will delete the counted objects with a lasting time less than the threshold. As tha statistic of our experiment results, the tracker with threshold=3 has the best accuracy.

## How to use the tracker?
configureable parameters:
- visualisation: turn on to see the 
- saveTrackerVisulisation: save the pictures with tracked bounding box as well as the object id.
- saveXml: Store the track statistic result into xml file.
- threshold: the threshold of a minimal lasting time to filter random false positive sequences, default is 3.
- inferenceMode: default is 'True', used for normal use case with inference results from detection model; set 'Faulse' to debug the tracker use random generated data.
### 1. Configure the path of inference labels and pictures:
- dir_path: This is the path to the folder you save the all segmented sequences.(The folders is named following the name rules)
- inferenceResultDir: The folder where all inference results are saved.
