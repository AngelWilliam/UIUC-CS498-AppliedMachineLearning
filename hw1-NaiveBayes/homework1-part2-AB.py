#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 18:02:40 2018

@author: mengyuxie
"""

import numpy as np  
import struct
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB  
from sklearn.metrics import accuracy_score
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
#from PIL import Image


def loadImageSet(filename):  
  
    binfile = open(filename, 'rb') 
    buffers = binfile.read()  
  
    head = struct.unpack_from('>IIII', buffers, 0)  
  
    offset = struct.calcsize('>IIII')   
    imgNum = head[1]  
    width = head[2]  
    height = head[3]  
  
    bits = imgNum * width * height   
    bitsString = '>' + str(bits) + 'B'   
  
    imgs_frame = struct.unpack_from(bitsString, buffers, offset)  
  
    binfile.close()  
    imgs = np.reshape(imgs_frame, [imgNum, width * height])  
  
    return imgs,head
  
  
def loadLabelSet(filename):  
  
    binfile = open(filename, 'rb')   
    buffers = binfile.read()  
  
    head = struct.unpack_from('>II', buffers, 0)   
  
    labelNum = head[1]  
    offset = struct.calcsize('>II')   
  
    numString = '>' + str(labelNum) + "B"  
    labels = struct.unpack_from(numString, buffers, offset) 
  
    binfile.close()  
    labels = np.reshape(labels, [labelNum])  
  
    return labels,head  
  


def resize_image(img,tol=0):
    # img is image data
    # tol  is tolerance
    arrylist=[]
    for item in img:
        single=np.reshape(item,[28,28])
        ##crop 
        mask = single>tol
        cropped=single[np.ix_(mask.any(1),mask.any(0))]
        resized=resize(cropped,(20,20))
        item = np.reshape(resized, [20 * 20])
        arrylist.append(item)        
    return np.array(arrylist)
    
'''partA-Naive beyes model'''
def training(model,train_x,train_y,test_x,test_y):
    clf = model
    clf.fit(train_x,train_y)
    pd=clf.predict(test_x)
    score=accuracy_score(test_y, pd)
    return score

'''partB-Decision tree model'''
def DecisionTree(depth,trees,train_x,train_y,test_x,test_y):
    clf = RandomForestClassifier(max_depth=depth, max_leaf_nodes=trees)
    clf.fit(train_x, train_y)
    pd=clf.predict(test_x)
    score=accuracy_score(test_y, pd)
    return score

  
if __name__ == "__main__":  
    train_x= 'train-images.idx3-ubyte'  
    train_y= 'train-labels.idx1-ubyte'  
    test_x='t10k-images.idx3-ubyte'
    test_y='t10k-labels.idx1-ubyte'
  
    train_imgs,train_data_head = loadImageSet(train_x)  
    test_imgs,test_data_head = loadImageSet(test_x)

    train_imgs_threshold=1*(train_imgs>128)
    test_imgs_threshold=1*(test_imgs>128)
    
    
    
    train_imgs_resize=resize_image(train_imgs)
    
    ##deal with the strange transformed output of resize()function
    avg=(np.amax(train_imgs_resize)+np.amin(train_imgs_resize))/2.0
    train_imgs_resize_threshold=1*(train_imgs_resize>avg)
    
    test_imgs_resize=resize_image(test_imgs)
    test_imgs_resize_threshold=1*(test_imgs_resize>avg)
    
    train_labels,train_labels_head = loadLabelSet(train_y)
    test_labels,test_labels_head = loadLabelSet(test_y)

    
    #plt.imshow(np.reshape(train_imgs_resize[2,:],[20,20]) , cmap='gray')
    
    
    '''partA-Naive beyes model-untouched'''
    score_gnb=training(GaussianNB(),train_imgs,train_labels,test_imgs,test_labels)
    score_bnb=training(BernoulliNB(),train_imgs,train_labels,test_imgs,test_labels)
    print(score_gnb,score_bnb)
    
    '''partA-Naive beyes model-resized with threshold'''
    score_gnb_resize=training(GaussianNB(),train_imgs_resize_threshold,train_labels,test_imgs_resize_threshold,test_labels)
    score_bnb_resize=training(BernoulliNB(),train_imgs_resize_threshold,train_labels,test_imgs_resize_threshold,test_labels)
    print(score_gnb_resize,score_bnb_resize)
    
    '''partB-Decision tree model'''
    depth_list=[4,8,16]
    trees_list=[10,20,30]
    for i in depth_list:
        for j in trees_list:
            print("resized")
            print(i,j,DecisionTree(i,j,train_imgs_resize_threshold,train_labels,test_imgs_resize_threshold,test_labels))
            print("untouched")
            print(i,j,DecisionTree(i,j,train_imgs,train_labels,test_imgs,test_labels))
    
 