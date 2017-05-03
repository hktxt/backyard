# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:21:50 2017

@author: Max
"""
#%% Evaluate one image
# when training, comment the following codes.

import numpy as np
import model
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(test_dir, num):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    images = []
    for file in os.listdir(test_dir):
        images.append(test_dir + file)
    n = len(images)
    print('There are', n, 'test images.')
    
    if num == 0:
        ind = np.random.randint(0, n)
        img_dir = images[ind]
    
        image = Image.open(img_dir)
        plt.imshow(image)
        image = image.resize([208, 208])
        image = np.array(image)
        return image
    elif num > n or num < 0:
        print("no such file, a random file will be tested.")
        ind = np.random.randint(0, n)
        img_dir = images[ind]
        image = Image.open(img_dir)
        plt.imshow(image)
        image = image.resize([208, 208])
        image = np.array(image)
        return image 
    else:
        img_dir = test_dir + str(num) + '.jpg'
        image = Image.open(img_dir)
        plt.imshow(image)
        image = image.resize([208, 208])
        image = np.array(image)
        return image

def evaluate_one_image(num):
    '''Test one image against the saved models and parameters
    '''
    
    # you need to change the directories to yours.
    test_dir = 'C:/cats_vs_dogs/test1/'
    image_array = get_one_image(test_dir,num)
    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
        
        # you need to change the directories to yours.
        logs_train_dir = 'C:/cats_vs_dogs/logs/train/'
                       
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a cat with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' %prediction[:, 1])

#%%
evaluate_one_image(100)