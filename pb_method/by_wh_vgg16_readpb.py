from urllib.request import urlretrieve
import os,shutil
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import cv2
import readimages as rd
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

save_path='./model.pb'
with tf.Session() as sess:
##    sess.run(tf.global_variables_initializer())
    with gfile.FastGFile(save_path,'rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def,name='')

    input_x=sess.graph.get_tensor_by_name('input_x:0')
    op=sess.graph.get_tensor_by_name('out/BiasAdd:0')
    
    test_data_path='E:/phython/github_program/vgg_16_class/testdata/test/'
    x_data,image_path=rd.load_images(test_data_path,(224,224))
    
    pre=tf.argmax(tf.nn.softmax(op),1)
    
    pre1=sess.run(pre,feed_dict={input_x:x_data})
  
