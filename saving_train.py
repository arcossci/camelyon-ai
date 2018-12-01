# saving_train.py
# In saving_train.py, we will need these functions from training_backend.py:
# •	load_image
# •	read_slide
# •	find_tissue_pixels
# •	apply_mask
# •	initialize_directories
# •	split_image_and_mask

# This is basically the first part of the function training in training_backend.py

# This function will take in a list of strings [‘img_001.tif, img_002.tif, etc.], 
# will download the corresponding images and masks, 
# and will do all the processing necessary to save the images and masks as jpgs. 
# We can probably keep the structure of the saved files the same for now
# (e.g. /data/img_001/tumor and  /data/img_001/no_tumor)

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
from skimage.color import rgb2gray
import cv2
import os
import random
import pathlib

def load_image(slide_path, tumor_mask_path):

    if not os.path.exists(slide_path):
        os.system('gsutil cp gs://terry-columbia/deep_learning_final_project/' + slide_path + ' ' + slide_path)
    if not  os.path.exists(tumor_mask_path):
        os.system('gsutil cp gs://terry-columbia/deep_learning_final_project/' + tumor_mask_path + ' ' + tumor_mask_path)

    slide = open_slide(slide_path)
    tumor_mask = open_slide(tumor_mask_path)

    for i in range(len(slide.level_dimensions)):
    
        assert tumor_mask.level_dimensions[i][0] == slide.level_dimensions[i][0]
        assert tumor_mask.level_dimensions[i][1] == slide.level_dimensions[i][1]

    # Verify downsampling works as expected
    width, height = slide.level_dimensions[7]
    assert width * slide.level_downsamples[7] == slide.level_dimensions[0][0]
    assert height * slide.level_downsamples[7] == slide.level_dimensions[0][1]
    
    return slide, tumor_mask

def read_slide(slide, x, y, level, width, height, as_float=False):
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im

def find_tissue_pixels(image, intensity=0.8):
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return zip(indices[0], indices[1])

def apply_mask(im, mask, color = 1):
    masked = np.zeros((im.shape[0], im.shape[1]))
    for x,y in mask: masked[x][y] = color
    return masked

## Define function to create directories to store slices of the training image
def initialize_directories(slide_path):
    BASE_DIR = os.getcwd()

    img_num = slide_path.split('_')[1].strip(".tif")

    # Create a folder for each image
    # Within that numbered image folder, create two other folders to hold the image slices 
    # that contain tumor cells and those that don't.

    DATA = 'data/'
    IMG_NUM_FOLDER = img_num + '/'
    TUMOR_FOLDER = 'tumor/'
    NO_TUMOR_FOLDER = 'no_tumor/'

    DATA_DIR = os.path.join(BASE_DIR, DATA)
    IMG_NUM_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER)
    TUMOR_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, TUMOR_FOLDER)
    NO_TUMOR_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, NO_TUMOR_FOLDER)

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(IMG_NUM_DIR):
        os.mkdir(IMG_NUM_DIR)
    if not os.path.exists(TUMOR_DIR):
        os.mkdir(TUMOR_DIR)
    if not os.path.exists(NO_TUMOR_DIR):
        os.mkdir(NO_TUMOR_DIR)

    # Return the locations of these folders for future use
    return DATA + IMG_NUM_FOLDER + TUMOR_FOLDER, DATA + IMG_NUM_FOLDER + NO_TUMOR_FOLDER

def split_image_and_mask(im, tumor_mask, tissue_mask,  num_pixels, level, slide_path):
    x,y = im.shape[0], im.shape[1]

    # Find the number of image slices that the original image will split into
    x_count, y_count = int(np.ceil(x / num_pixels)), int(np.ceil(y/num_pixels))

    tumor_folder, no_tumor_folder = initialize_directories(slide_path)


    try:
        for i in range(x_count):
            for j in range(y_count):
                im_slice = np.zeros((num_pixels, num_pixels, 3))
                tissue_mask_slice = np.zeros((num_pixels, num_pixels))
                tumor_mask_slice = np.zeros((num_pixels, num_pixels))

                # Name the image slice based on where it lies in the original image
                string_name = 'img_level%d_'%(level) + str(i * y_count + j)

                # Logic to handle end conditions
                if i == x_count-1:
                    ub_x = x
                    assign_x = x - (x_count-1)*num_pixels
                else:
                    ub_x = (i+1) * num_pixels
                    assign_x = num_pixels

                if j == y_count-1:
                    ub_y = y
                    assign_y = y - (y_count-1)*num_pixels
                else:
                    ub_y = (j+1) * num_pixels
                    assign_y = num_pixels

                    # Assign the pixels to the slice of the tissue mas
                tissue_mask_slice[0:assign_x, 0:assign_y] = tissue_mask[(i*num_pixels) :ub_x, (j * num_pixels) :ub_y]

                try:
                    # If 70% of the slice is tissue, process with classifying the slice
                    # (tumor/no tumor) and saving 
                    if np.mean(tissue_mask_slice) > 0.7:
                        im_slice[0:assign_x, 0:assign_y, :] = im[(i*num_pixels) :ub_x, (j * num_pixels) :ub_y, :]
                        tumor_mask_slice[0:assign_x, 0:assign_y] = tumor_mask[(i*num_pixels) :ub_x, (j * num_pixels) :ub_y]

                        if np.max(tumor_mask_slice) > 0:
                            im_file_name = tumor_folder + string_name + ".jpg"
                        else:
                            im_file_name = no_tumor_folder + string_name + ".jpg"

                        cv2.imwrite(im_file_name, im_slice)
                except Exception as oerr:
                    print('Error with saving:', oerr)
    except Exception as oerr:
        print('Error with slicing:', oerr)

def train_part_1(training_image_path_list, num_pixels = 64, num_level = 2):

	for i in training_image_path_list:
	    slide_path = 'tumor_091.tif'
	    tumor_mask_path = i.split('.')[0]+'_mask.tif'

	    slide, tumor_mask  = load_image(slide_path, tumor_mask_path)
	    width, height = slide.level_dimensions[num_level][0], slide.level_dimensions[num_level][1]
	        
	    slide = read_slide(slide,
	                             x=0,
	                             y=0,
	                             level=num_level,
	                             width=width,
	                             height=height)


	    tumor_mask = read_slide(tumor_mask,
	                            x=0,
	                            y=0,
	                            level=num_level,
	                            width=width,
	                            height=height)

	    ## Convert the mask from RGB to a black/white binary
	    tumor_mask = tumor_mask[:,:,0]

	    ## Determine the portions of the image that are tissue
	    tissue_pixels = list(find_tissue_pixels(slide))

	    ## Turn the tissue pixels into a mask
	    tissue_regions = apply_mask(slide, tissue_pixels)

	    ## Call the split function on the training data
	    split_image_and_mask(slide, tumor_mask, tissue_regions, num_pixels, num_level, slide_path)














