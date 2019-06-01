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

learning_rate=0.001
num_steps=10
batch_size=20
disp_step=2
dropout=0.75
N_CLASSES=2


vgg_mean=[103.939,116.779,123.68]
data_dict=np.load('E:/phython/tensorflow-test/read image/test/vgg16.npy',encoding='latin1').item()
train_data_path='E:/phython/github_program/vgg_16_class/testdata'



tfx=tf.placeholder(tf.float32,[None,224,224,3],name='input_x')
tfy=tf.placeholder(tf.int32,[None,N_CLASSES],name='input_x')

def vgg(tfx):
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=tfx * 255.0)
    bgr = tf.concat(axis=3, values=[
        blue -vgg_mean[0],
        green - vgg_mean[1],
        red -vgg_mean[2],
    ])

    def max_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, data_dict[name][1]))
            return lout
    conv1_1 =conv_layer(bgr, "conv1_1")
    conv1_2 =conv_layer(conv1_1, "conv1_2")
    pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 =conv_layer(pool1, "conv2_1")
    conv2_2 =conv_layer(conv2_1, "conv2_2")
    pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 =conv_layer(pool2, "conv3_1")
    conv3_2 =conv_layer(conv3_1, "conv3_2")
    conv3_3 =conv_layer(conv3_2, "conv3_3")
    pool3 = max_pool(conv3_3, 'pool3')

    conv4_1 =conv_layer(pool3, "conv4_1")
    conv4_2 =conv_layer(conv4_1, "conv4_2")
    conv4_3 =conv_layer(conv4_2, "conv4_3")
    pool4 = max_pool(conv4_3, 'pool4')

    conv5_1 =conv_layer(pool4, "conv5_1")
    conv5_2 =conv_layer(conv5_1, "conv5_2")
    conv5_3 =conv_layer(conv5_2, "conv5_3")
    pool5 = max_pool(conv5_3, 'pool5')

    flatten = tf.reshape(pool5, [-1, 7*7*512])
    fc6 = tf.layers.dense( flatten, 10, tf.nn.relu, name='fc6',reuse=tf.AUTO_REUSE)
    out = tf.layers.dense( fc6, N_CLASSES, name='out',reuse=tf.AUTO_REUSE)
    return out

save_path='./model.pb'
def train(train_data_path):
    x_data,y_data=rd.read_images(train_data_path,batch_size,(224,224))
    out=vgg(tfx)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( 
            logits=out, labels=tfy))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)
    
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_steps):
            los,_=sess.run([loss_op,train_op],feed_dict={tfx:x_data,tfy:y_data})
            if i%disp_step==0:
                print(i,los)
        
        constant_graph=graph_util.convert_variables_to_constants(sess,sess.graph_def,['out/BiasAdd','input_x'])
        with tf.gfile.FastGFile(save_path,'wb') as f:
            f.write(constant_graph.SerializeToString())
        
train(train_data_path)  
