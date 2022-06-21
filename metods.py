import os
import cv2
import random
import numpy as np

from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from matplotlib import pyplot as plt
import numpy as np
import skimage.measure

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def readImages(path,  IMG_HEIGHT = 128, IMG_WIDTH = 128, is_gray=0):
    cv2_image = cv2.imread(path, is_gray)
    print(path)
    cv2_image = cv2.resize(cv2_image, (IMG_HEIGHT, IMG_WIDTH))
    cv2.imwrite('img.jpg', cv2_image)
    
    img = cv2.imread('img.jpg', is_gray)
    os.remove('img.jpg')
    
    return img


def fillHoles(pred_img):
    pred_img = (pred_img/255.)* 255.0
    dilated = cv2.dilate(pred_img.copy(), None, iterations=3) #gray img
    eroded = cv2.erode(dilated.copy(), None, iterations=3) #gray img
    _ , binary_img = cv2.threshold(eroded, 120, 255, cv2.THRESH_BINARY) #binary img
    
    return binary_img


def removeSmallObjects(img, min_area = 200):
    labeled_image, _ = skimage.measure.label(img, connectivity=2, return_num=True)
    
    # compute object features and extract object areas
    object_features = skimage.measure.regionprops(labeled_image)
    object_areas = [objf["area"] for objf in object_features]
    object_areas
    
    min_area = min_area
    large_objects = []
    for objf in object_features:
        if objf["area"] > min_area:
            large_objects.append(objf["label"])
    # print("Found", len(large_objects), "objects!")
    
    for _, objf in enumerate(object_features, start=1):
        if objf["area"] < min_area:
            labeled_image[labeled_image == objf["label"]] = 0
    
    colored_label_image = skimage.color.label2rgb(labeled_image, bg_label=0)
    
    colorChannel1 = colored_label_image[:,:,0]
    colorChannel2 = colored_label_image[:,:,2]
    combinedChannels = colorChannel1 + colorChannel2
    return combinedChannels


def showRandomPredict(images, masks, predicts, sample_count, if_predicted= False, img_size =(128, 128)):
    sample_count = sample_count * 4
    plt.figure(figsize=(15, 30))
    for i in range(0, sample_count, 4):
        img_index = random.choice(range(len(images)))
        plt.subplot(5,4,i + 1)
        random_img = images[img_index,:,:]
        random_img = cv2.resize(random_img, img_size)
        plt.imshow(random_img, cmap=plt.cm.bone)
        plt.axis('off')
        plt.title('Lung X-Ray')

        plt.subplot(5,4,i + 2)
        random_mask = masks[img_index,:,:]
        random_mask = cv2.resize(random_mask, img_size)
        plt.imshow(random_mask, cmap='gray')
        plt.axis('off')
        plt.title('Mask Ground Truth')
        
        if not if_predicted:
            continue
        random_pred = predicts[img_index,:,:,0]
        plt.subplot(5,4,i + 3)
        plt.imshow(random_pred, cmap='gray')
        plt.axis('off')
        plt.title('Predicted Mask')

        plt.subplot(5,4,i + 4)
        plt.imshow(cv2.bitwise_and(images[img_index,:,:], images[img_index,:,:], mask=random_pred.astype(np.uint8)), cmap=plt.cm.bone)
        plt.axis('off')
        plt.title('Predicted Lung Segmentation')
    plt.show()
    
    