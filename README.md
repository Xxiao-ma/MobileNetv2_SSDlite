# mobilenet_v2_ssdlite_keras
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
MobileNetv2 + SSDlite:
notebook: mobileNetv2_ssdLite.ipynb

# Bottleneck LSTM:
MobileNetv2 drop 19*19 and 1*1 feature map + bottleneckLSTM layer + SSDlite
notebook: mobileNetv2_ssdLite_bottleneckLstm.ipynb
