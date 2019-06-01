import tensorflow as tf
import os
import cv2
import numpy as np

##data_path='E:/phython/tensorflow-test/transfer learning/data'

n_class=2

image_h=64
image_w=64
image_channel=3


def read_images(data_path,batch_size,img_shape):
    imagepaths,labels=list(),list()
    label=0
    classes=sorted(os.walk(data_path).__next__()[1])
    for c in classes:
        c_dir=os.path.join(data_path,c)
        walk=os.walk(c_dir).__next__()
        for sample in walk[2]:
            if sample.endswith('.jpg'):
                imagepaths.append(os.path.join(c_dir,sample))
                labels.append(label)
        label+=1
        
    length=len(labels)
    batch=np.random.choice(np.array(length),batch_size)
    x_batch=[]
    y_batch=[]
    for i in batch:
        img=cv2.imread(imagepaths[i])
        img=img/255.0
        image=cv2.resize(img,img_shape)
        x_batch.append(image)
##        y_batch.append(labels[i])
        if labels[i]==0:
            y_batch.append([1.0,0.0])
        else:
            y_batch.append([0.0,1.0])

    y_batch=np.reshape(y_batch,[batch_size,2])
    
    return x_batch,y_batch

##data_path='E:/phython/github_program/vgg_16_class/traindata'
##img_shape=(224,224)

def load_images(data_path,img_shape):
    imagepaths=list()
    classes=sorted(os.walk(data_path).__next__()[1])
    for c in classes:
        c_dir=os.path.join(data_path,c)
        walk=os.walk(c_dir).__next__()
        for sample in walk[2]:
            
            if sample.endswith('.jpg'):
                
                imagepaths.append(os.path.join(c_dir,sample))
           
    x_data=list()

    for im in imagepaths:
        img=cv2.imread(im)
        img=img/255.0
        image=cv2.resize(img,img_shape)
        x_data.append(image)
    return x_data,imagepaths
    
test_data_path='E:/phython/github_program/vgg_16_class/testdata/test'
a,imagepaths=load_images(test_data_path,(224,224))

