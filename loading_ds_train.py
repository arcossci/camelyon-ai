# loading_ds_train.py
# In loading_ds_train.py we will need the following functions from training_backend.py
# •    gen_image_paths
# •    preprocess_images
# •    load_and_preprocess_image
# •    create_tf_dataset

# This is basically the second part of the function training in training_backend.py

# This function will take in a list of strings [‘img_001.tif, img_002.tif, etc.] and will 
# output a single datastream with paths to the training images we want to use
# (should maintain equal tumor/no_tumor proportions). It will also output the steps per epoch.

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
from skimage.color import rgb2gray
import cv2
import os
import random
import pathlib
import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
from skimage.color import rgb2gray
import cv2
import os
import random
import pathlib

def gen_image_paths(training_image_path_list, num_level):
    all_images_image_paths = []
    all_images_image_labels = []

    for i in training_image_path_list:

        slide_path = i

        img_num = slide_path.split('_')[1].strip(".tif")

        data_root_tumor = pathlib.Path('data/' + img_num + '/level_' + str(num_level) + '/tumor')
        all_image_paths_tumor = list(data_root_tumor.glob('*'))
        num_tumor_images = len(all_image_paths_tumor)
        
        data_root_notumor = pathlib.Path('data/' + img_num + '/level_' + str(num_level) + '/no_tumor')
        all_image_paths_notumor = list(data_root_notumor.glob('*'))
        random.shuffle(all_image_paths_notumor)
        all_image_paths_notumor = all_image_paths_notumor[0:num_tumor_images]

        all_image_paths = [str(path) for path in all_image_paths_tumor + all_image_paths_notumor]
        random.shuffle(all_image_paths)

        data_root = pathlib.Path('data/' + img_num + '/level_' + str(num_level))
        label_names = sorted(item.name for item in data_root.glob('*') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))

        all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in all_image_paths]

        #update all image path lists
        all_images_image_paths = all_images_image_paths + all_image_paths
        all_images_image_labels = all_images_image_labels + all_image_labels
    
    return all_images_image_paths, all_images_image_labels

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [128, 128])
    image /= 255.0  # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)

def create_tf_dataset(all_image_paths_1, all_image_paths_2, all_image_labels):
    
    path_ds_1 = tf.data.Dataset.from_tensor_slices(all_image_paths_1)
    image_ds_1 = path_ds_1.map(load_and_preprocess_image, num_parallel_calls=8)
    
    path_ds_2 = tf.data.Dataset.from_tensor_slices(all_image_paths_2)
    image_ds_2 = path_ds_2.map(load_and_preprocess_image, num_parallel_calls=8)
    
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    
    image_label_ds = tf.data.Dataset.zip(((image_ds_1,image_ds_2), label_ds))

    BATCH_SIZE = 4

    steps_per_epoch = int(np.ceil(len(all_image_paths_1)/BATCH_SIZE))

    # Setting a shuffle buffer size larger than the dataset ensures that the data is completely shuffled.
    ds = image_label_ds.repeat()
    ds = ds.shuffle(buffer_size=4000)
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetches batches, asynchronously while the model is training.
    ds = ds.prefetch(1)


    return ds, steps_per_epoch


def train_part_2(training_image_path_list, num_level_1, num_level_2):

    # change input here from a specific image to an image path
    all_image_paths_1, all_image_labels_1 = gen_image_paths(training_image_path_list, num_level_1)
    
    # create the second file path to mimic the 1st
    all_image_paths_2 = []
    for i in all_image_paths_1:
        split_str = i.split('/')
        split_str[2] = 'level_'+str(num_level_2)
        path_2_string = ''
        for j in split_str:
            if j == split_str[-1]:
                path_2_string = path_2_string + j
            else:
                path_2_string = path_2_string + j + '/'
        all_image_paths_2.append(path_2_string)

   
    
    ## Create tf.Dataset for training
    ds, steps_per_epoch = create_tf_dataset(all_image_paths_1, all_image_paths_2, all_image_labels_1)

    ## Posibly downscale slide
    
    ## Return training data
    return ds, steps_per_epoch