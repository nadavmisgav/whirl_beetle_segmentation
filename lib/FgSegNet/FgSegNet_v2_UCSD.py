#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 2018

@author: longang
"""

import os
import random as rn
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# set current working directory
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

# =============================================================================
#  For reprodocable results
# =============================================================================
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from tensorflow.python.keras import backend as K

tf.compat.v1.set_random_seed(1234)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

import glob

import keras
from keras.preprocessing import image as kImage
from keras.utils.data_utils import get_file
from sklearn.utils import compute_class_weight

from FgSegNet_v2_module import FgSegNet_v2_module

# alert the user
if keras.__version__!= '2.0.6' or tf.__version__!='1.1.0' or sys.version_info[0]<3:
    print('We implemented using [keras v2.0.6, tensorflow-gpu v1.1.0, python v3.6.3], other versions than these may cause errors somehow!\n')

# =============================================================================
# Few frames, load into memory directly
# =============================================================================
def getData(data_path: Path, label_path: Path):
    X_list = list(data_path.glob("*.jpg"))
    Y_list = list(label_path.glob("*.jpg"))

    if len(Y_list)<=0 or len(X_list)<=0:
        raise ValueError('System cannot find the dataset path or ground-truth path. Please give the correct path.')
    
    if len(X_list)!=len(Y_list):
        raise ValueError('The number of X_list and Y_list must be equal.')
        
    # X must be corresponded to Y
    X_list = sorted(X_list)
    Y_list = sorted(Y_list)
    
    # process training images
    X = []
    Y = []
    for i in range(0, len(X_list)):
        x = kImage.load_img(X_list[i])
        x = kImage.img_to_array(x)
        X.append(x)
        
        x = kImage.load_img(Y_list[i], grayscale = True)
        x = kImage.img_to_array(x)
        x /= 255.0
        x = np.floor(x)
        Y.append(x)
        
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Shuffle the training data
    idx = list(range(X.shape[0]))
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    
    # compute class weights
    cls_weight_list = []
    for i in range(Y.shape[0]):
        # y = Y[i].reshape(-1)
        lb = np.unique(Y[i].reshape(-1)) #  0., 1
        cls_weight = compute_class_weight(class_weight  = 'balanced', classes  = lb , y =  Y[i].reshape(-1))
        class_0 = cls_weight[0]
        class_1 = cls_weight[1] if len(lb)>1 else 1.0
        
        cls_weight_dict = {0:class_0, 1: class_1}
        cls_weight_list.append(cls_weight_dict)
    # del y
    cls_weight_list = np.asarray(cls_weight_list)

    return [X, Y, cls_weight_list]
    
def train(data, mdl_path, vgg_weights_path, max_epoch: int, lr: float):
    
    ### hyper-params
    val_split = 0.2
    batch_size = 1
    ###
    
    img_shape = data[0][0].shape #(height, width, channel)
    scene = ""
    model = FgSegNet_v2_module(lr, img_shape, scene, vgg_weights_path)
    model = model.initModel('UCSD')

    # make sure that training input shape equals to model output
    input_shape = (img_shape[0], img_shape[1])
    output_shape = (model.output._keras_shape[1], model.output._keras_shape[2])
    assert input_shape==output_shape, 'Given input shape:' + str(input_shape) + ', but your model outputs shape:' + str(output_shape)
    
    early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1, mode='auto')  
    redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')
    model.fit(data[0], data[1], 
              validation_split=val_split,
              epochs=max_epoch, batch_size=batch_size, 
              callbacks=[redu, early], verbose=1, class_weight=data[2], shuffle = True)
    
    model.save(mdl_path)
    del model, data, early, redu


def main(data_path: Path, label_path: Path, epochs: int, lr: float):
    vgg_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    if not os.path.exists(vgg_weights_path):
        WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        vgg_weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP, cache_subdir='models',
                                    file_hash='6d6bbae143d832006294945121d1f1fc')
    
    mdl_path = Path(__file__).parents[0].parent / "weights" / f"{int(time.time())}.h5"
    
    results = getData(data_path, label_path)
    train(results, str(mdl_path), vgg_weights_path, epochs, lr)
    print(f"Saved model in {mdl_path.absolute()}")
