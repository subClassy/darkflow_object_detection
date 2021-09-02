from darkflow.net.build import TFNet
import cv2
import tensorflow as tf

config = tf.compat.v1.ConfigProto(log_device_placement = False)
config.gpu_options.allow_growth = False 

with tf.compat.v1.Session(config=config) as sess:
    options = {
                'model': './cfg/yolo.cfg',
                'load': './data/yolov2.weights',
                'threshold': 0.6
               }
    tfnet = TFNet(options)    

