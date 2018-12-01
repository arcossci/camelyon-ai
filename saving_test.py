# saving_test.py
# testing_backend.py  1) 	saving_test.py   &	2) loading_ds_test.py 

# In saving_tset.py, we will need these functions from training_backend.py:
# •	load_image
# •	read_slide
# •	find_tissue_pixels
# •	apply_mask

# And the following functions from testing_backend.py:
# •	initialize_directories_test
# •	split_image_test

# This is basically the first part of the function testing in testing_backend.py

# Similarly, this function will take in a list of strings
# [‘img_003.tif, img_004.tif, etc.],
# will download the corresponding images and masks,
# and will do all the processing necessary to save the test images as jpgs.
# We can probably keep the structure of the saved files the same for now
# (e.g. /data/img_003/tissue_only) but only worry about saving the tissue
# only testing sample (not all image splices).

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

## Function to split and save test images, slightly different than analogous
## function for training
def split_image_test(im, tissue_mask, num_pixels, level_num, slide_path):
    x, y = im.shape[0], im.shape[1]
    x_count, y_count = int(np.ceil(x / num_pixels)), int(np.ceil(y / num_pixels))

    tissue_folder, all_folder = initialize_directories_test(slide_path)

    try:
        for i in range(x_count):
            for j in range(y_count):
                im_slice = np.zeros((num_pixels, num_pixels, 3))
                im_tissue_slice = np.zeros((num_pixels, num_pixels, 3))
                tissue_mask_slice = np.zeros((num_pixels, num_pixels))

                string_name = 'img_level%d_' % (level_num) + str(i * y_count + j)

                if i == x_count - 1:
                    ub_x = x
                    assign_x = x - (x_count - 1) * num_pixels
                else:
                    ub_x = (i + 1) * num_pixels
                    assign_x = num_pixels

                if j == y_count - 1:
                    ub_y = y
                    assign_y = y - (y_count - 1) * num_pixels
                else:
                    ub_y = (j + 1) * num_pixels
                    assign_y = num_pixels

                tissue_mask_slice[0:assign_x, 0:assign_y] = tissue_mask[(i * num_pixels):ub_x, (j * num_pixels):ub_y]

                try:
                    if np.mean(tissue_mask_slice) > 0.7:
                        im_tissue_slice[0:assign_x, 0:assign_y, :] = im[(i * num_pixels):ub_x, (j * num_pixels):ub_y, :]
                        im_file_name_tissue = tissue_folder + string_name + ".jpg"
                        cv2.imwrite(im_file_name_tissue, im_tissue_slice)

                    im_slice[0:assign_x, 0:assign_y, :] = im[(i * num_pixels):ub_x, (j * num_pixels):ub_y, :]
                    im_file_name_all = all_folder + string_name + ".jpg"
                    cv2.imwrite(im_file_name_all, im_slice)

                except Exception as oerr:
                    print('Error with saving:', oerr)

    except Exception as oerr:
        print('Error with slicing:', oerr)


#def test_part_1(testing_image_path_list, num_pixels=64, num_level=2):
def test_part_1(testing_image_path, num_pixels=64, num_level=2):

#def testing(num_pixels, num_level):
	# for i in testing_image_path_list:
    #slide_path_test = i
    slide_path_test = testing_image_path
    tumor_mask_path_test = i.split('.')[0]+'_mask.tif'
    print(slide_path_test, tumor_mask_path_test)

    ## Retrieve slide parameters before overwriting
    slide, tumor_mask = load_image(slide_path_test, tumor_mask_path_test)
    width, height = slide.level_dimensions[num_level][0], slide.level_dimensions[num_level][1]
    
    
    ## Read training image at slide level 3
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

    ## Retrieve new array dimensions
    image_depth, image_width = int(np.ceil(slide.shape[0] / num_pixels)), int(np.ceil(slide.shape[1] / num_pixels))
    
    ## Convert the mask from RGB to a black/white binary
    tumor_mask = tumor_mask[:, :, 0]

    ## Determine the portions of the image that are tissue
    
    tissue_pixels = list(find_tissue_pixels(slide))

    ## Turn the tissue pixels into a mask
    tissue_regions = apply_mask(slide, tissue_pixels)

    split_image_test(slide, tissue_regions, num_pixels, num_level, slide_path_test)

    return image_depth, image_width, tumor_mask, tissue_regions, slide

