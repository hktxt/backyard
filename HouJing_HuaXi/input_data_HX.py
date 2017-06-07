# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 18:30:32 2017

@author: Max
"""

import tensorflow as tf
import numpy as np
import os
import math

#%%

# you need to change this to your data directory
file_dir = 'D:/DATA/huaxi/data/train/'

def get_files(file_dir, ratio):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    XSW = [] 
    label_XSW = []
    HY = []
    label_HY = []
    BB = [] 
    label_BB = []
    XJ = []
    label_XJ = []
    ZC = [] 
    label_ZC = []

    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[1]=='1':
            XSW.append(file_dir + file)
            label_XSW.append(0)
        elif name[1]=='2':
            HY.append(file_dir + file)
            label_HY.append(1)
        elif name[1]=='3':
            BB.append(file_dir + file)
            label_BB.append(2)
        elif name[1]=='4':
            XJ.append(file_dir + file)
            label_XJ.append(3)
        else:
            ZC.append(file_dir + file)
            label_ZC.append(4)
    print('There are %d XSW\nThere are %d HY\nThere are %d BB\nThere are %d XJ\nThere are %d ZC' %(len(XSW), len(HY), len(BB), len(XJ), len(ZC)))
    
    image_list = np.hstack((XSW, HY, BB, XJ, ZC))
    label_list = np.hstack((label_XSW, label_HY, label_BB, label_XJ, label_ZC))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)   
    
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
#    all_image_list = temp[:, 0]
#    all_label_list = temp[:, 1]
#    
    
    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample*ratio) # number of validation samples
    n_train = n_sample - n_val # number of trainning samples
    
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
    
    
    #return all_image_list, all_label_list
    return tra_images,tra_labels,val_images,val_labels
    #return temp,all_image_list,tra_images


#%%

def get_batch(image, label, image_W, image_H, batch_size, n_classes, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)    
    # if you want to test the generated batches of images, you might want to comment the following line.
    
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    label_batch = tf.one_hot(label_batch, depth= n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])
    #label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch
