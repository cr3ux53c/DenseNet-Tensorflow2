"""Dataset loader"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
from os.path import join
import cv2
import os

class pneumonia:
    @staticmethod
    def load_data(dataset_dir):

        sr_size = 256
        sr_size = 1000
        sr_size = 700
        sr_size = 224
        scale = 1
        model_postfix = '_mod3_224'
        model_postfix = ''

        train_normal = glob.glob(join(dataset_dir, 'train_NORMAL_' + str(sr_size) + model_postfix, '*.png'))[:]
        train_abnormal = glob.glob(join(dataset_dir, 'train_PNEUMONIA_' + str(sr_size) + model_postfix, '*.png'))[:]
        test_normal = glob.glob(join(dataset_dir, 'test_NORMAL_' + str(sr_size) + model_postfix, '*.png'))[:]
        test_abnormal = glob.glob(join(dataset_dir, 'test_PNEUMONIA_' + str(sr_size) + model_postfix, '*.png'))[:]

        if os.path.isfile(join(dataset_dir, 'train_' + str(sr_size)+'.npy')):
            x_train = np.load(join(dataset_dir, 'train_' + str(sr_size)+'.npy'))
            x_test = np.load(join(dataset_dir, 'test_' + str(sr_size)+'.npy'))
        else:
            x_train = np.zeros((0, 224, 224, 3))
            for index, train_path in enumerate(train_normal + train_abnormal):
                x_train = np.concatenate((x_train, np.expand_dims(cv2.imread(train_path), axis=0)), axis=0)
                print('Load train dataset sample #{}/{}, {}'.format(index, len(train_normal + train_abnormal)-1, os.path.basename(train_path)))
            
            x_test = np.zeros((0, 224, 224, 3))
            for index, test_path in enumerate(test_normal + test_abnormal):
                x_test = np.concatenate((x_test, np.expand_dims(cv2.imread(test_path), axis=0)), axis=0)
                print('Load test dataset sample #{}/{}, {}'.format(index, len(test_normal + test_abnormal)-1, os.path.basename(test_path)))

            np.save(join(dataset_dir, 'train_' + str(sr_size)+'.npy'), x_train)
            np.save(join(dataset_dir, 'test_' + str(sr_size)+'.npy'), x_test)

        # Divice x_train to x_train and x_val
        y_train = np.append(np.zeros((len(train_normal), 1), dtype=np.int64), np.ones((len(train_abnormal), 1), dtype=np.int64))
        y_test = np.append(np.zeros((len(test_normal), 1), dtype=np.int64), np.ones((len(test_abnormal), 1), dtype=np.int64))

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, shuffle=True, stratify=y_train, random_state=None)

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def load(dataset_name, dataset_dir=None, datagen_flow=False,
         weight_classes=False, batch_size=32,
         rotation_range = 10, width_shift_range = 0.10,
         height_shift_range = 0.10, horizontal_flip = True,
         train_size=None, test_size=None):
    """
    Load specific dataset.

    Args:
        dataset_name (str): name of the dataset.

    Returns (train_gen, val_gen, test_gen, nb_classes, image_shape, class_weights):.
    """

    if dataset_name == "cifar10":
        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_val, y_val = x_test, y_test
    elif dataset_name == "cifar100":
        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
        x_val, y_val = x_test, y_test
    elif dataset_name == "pneumonia":
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = pneumonia.load_data(dataset_dir=dataset_dir)
    else:
        raise ValueError("Unknow dataset: {}".format(dataset_name))

    image_shape = np.shape(x_train)[1:]

    nb_classes = len(np.unique(y_train))

    class_weights = None
    if weight_classes:
        class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
    
    train_datagen_args = dict(featurewise_center=True,
                              featurewise_std_normalization=True,
                              rotation_range=rotation_range,
                              width_shift_range=width_shift_range,
                              height_shift_range=height_shift_range,
                              horizontal_flip=horizontal_flip,
                              fill_mode='constant',
                              cval=0)
    train_datagen = ImageDataGenerator(train_datagen_args)
    train_datagen.fit(x_train)

    test_datagen_args = dict(featurewise_center=True,
                            featurewise_std_normalization=True,
                            fill_mode='constant',
                            cval=0)
    test_datagen = ImageDataGenerator(test_datagen_args)
    test_datagen.fit(x_train)

    val_datagen = ImageDataGenerator(test_datagen_args)
    val_datagen.fit(x_train)

    train = (train_datagen, train_datagen_args, len(x_train), len(y_train))
    val = (val_datagen, test_datagen_args, len(x_val), len(y_val))
    test = (test_datagen, test_datagen_args, len(x_test), len(y_test))

    if datagen_flow:
        # create data generators
        train_gen = train_datagen.flow(x_train, y_train, batch_size=batch_size)
        val_gen = val_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)
        test_gen = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

        train = (train_gen, len(x_train), len(y_train))
        val = (val_gen, len(x_val), len(y_val))
        test = (test_gen, len(x_test), len(y_test))

    return train, val, test, nb_classes, image_shape, class_weights
