import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
from skimage.color import rgb2gray
import cv2
import os
import random
import pathlib

def initialize_directories_test(slide_path, level):
    BASE_DIR = os.getcwd()

    img_num = slide_path.split('_')[1].strip(".tif")

    DATA = 'data/'
    IMG_NUM_FOLDER = img_num + '/'
    LEVEL_FOLDER = 'level_'+str(level)+'/'
    TISSUE_FOLDER = 'tissue_only/'
    ALL_FOLDER = 'all/'

    DATA_DIR = os.path.join(BASE_DIR, DATA)
    IMG_NUM_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER)
    LEVEL_NUM_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER)
    TISSUE_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER, TISSUE_FOLDER)
    ALL_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER, ALL_FOLDER)

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(IMG_NUM_DIR):
        os.mkdir(IMG_NUM_DIR)
    if not os.path.exists(LEVEL_NUM_DIR):
        os.mkdir(LEVEL_NUM_DIR)
    if not os.path.exists(TISSUE_DIR):
        os.mkdir(TISSUE_DIR)
    if not os.path.exists(ALL_DIR):
        os.mkdir(ALL_DIR)

    return DATA + IMG_NUM_FOLDER + LEVEL_FOLDER + TISSUE_FOLDER, DATA + IMG_NUM_FOLDER + LEVEL_FOLDER + ALL_FOLDER

def load_image(slide_path, tumor_mask_path):

    if not os.path.exists(slide_path):
        os.system('gsutil cp gs://terry-columbia/deep_learning_final_project/' + slide_path + ' ' + slide_path)
    if not  os.path.exists(tumor_mask_path):
        os.system('gsutil cp gs://terry-columbia/deep_learning_final_project/' + tumor_mask_path + ' ' + tumor_mask_path)

    slide = open_slide(slide_path)
    tumor_mask = open_slide(tumor_mask_path)

    for i in range(len(slide.level_dimensions)-1):
        
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

    tissue_folder, all_folder = initialize_directories_test(slide_path, level_num)

    try:
        for i in range(x_count):
            for j in range(y_count):
                im_slice = np.zeros((num_pixels, num_pixels, 3))
                im_tissue_slice = np.zeros((num_pixels, num_pixels, 3))
                tissue_mask_slice = np.zeros((num_pixels, num_pixels))

                string_name = 'img_' + str(i * y_count + j)

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

def load_second_level(slide_path, input_level, num_input_pixels, output_level, num_output_pixels):
    img_num = slide_path.split('_')[1].strip(".tif")

    BASE_DIR = os.getcwd()
    DATA = 'data/'
    LEVEL_INPUT_FOLDER = 'level_' + str(input_level) + '/'
    LEVEL_OUTPUT_FOLDER = 'level_' + str(output_level) + '/'
    TISSUE_FOLDER = 'tissue_only/'
    ALL_FOLDER = 'all/'

    TISSUE_DIR_INPUT   = os.path.join(BASE_DIR, DATA, img_num, LEVEL_INPUT_FOLDER, TISSUE_FOLDER)
    ALL_DIR_INPUT      = os.path.join(BASE_DIR, DATA, img_num, LEVEL_INPUT_FOLDER, ALL_FOLDER)
    LEVEL_DIR_OUTPUT   = os.path.join(BASE_DIR, DATA, img_num, LEVEL_OUTPUT_FOLDER)
    TISSUE_DIR_OUTPUT  = os.path.join(BASE_DIR, DATA, img_num, LEVEL_OUTPUT_FOLDER, TISSUE_FOLDER)
    ALL_DIR_OUTPUT     = os.path.join(BASE_DIR, DATA, img_num, LEVEL_OUTPUT_FOLDER, ALL_FOLDER)


    if not os.path.exists(LEVEL_DIR_OUTPUT):
        os.mkdir(LEVEL_DIR_OUTPUT)
    if not os.path.exists(TISSUE_DIR_OUTPUT):
        os.mkdir(TISSUE_DIR_OUTPUT)
    if not os.path.exists(ALL_DIR_OUTPUT):
        os.mkdir(ALL_DIR_OUTPUT)


    data_root_tissue_input = pathlib.Path(TISSUE_DIR_INPUT)
    all_image_paths_tissue_input = list(data_root_tissue_input.glob('*'))
    all_paths_tissue_str_input   = [str(path) for path in all_image_paths_tissue_input]
    num_tissue_images_input = len(all_image_paths_tissue_input)
    
    data_root_all_input = pathlib.Path(ALL_DIR_INPUT)
    all_image_paths_all_input = list(data_root_all_input.glob('*'))
    all_paths_all_str_input   = [str(path) for path in all_image_paths_all_input]
    num_all_images_input = len(all_paths_all_str_input)

    slide = open_slide(slide_path)
    input_width, input_height = slide.level_dimensions[input_level][0], slide.level_dimensions[input_level][1]
    output_width, output_height = slide.level_dimensions[output_level][0], slide.level_dimensions[output_level][1]
    slide = read_slide(slide,
                                 x=0,
                                 y=0,
                                 level=output_level,
                                 width=output_width,
                                 height=output_height)

    ## Find number of images that can fit in x and y direction of input given input number of pixels
    row_count, col_count = int(np.ceil(input_height / num_input_pixels)), int(np.ceil(input_width / num_input_pixels))

    for i in all_paths_tissue_str_input + all_paths_all_str_input:
        try:

            img_index = i.split('_')[-1].strip(".jpg")


            start_input_row_count, start_input_col_count = (int(img_index) // col_count), (int(img_index) %  col_count)
            start_input_row_index = start_input_row_count * num_input_pixels 
            start_input_col_index = start_input_col_count * num_input_pixels

            scale_factor = 2 ** (output_level - input_level)
            shift_middle = (1 - 1/scale_factor)/2

            start_output_row = start_input_row_index * 1/scale_factor
            start_output_col = start_input_col_index * 1/scale_factor

            shift_pixels_up_left = shift_middle * num_output_pixels
            shift_pixels_down_right = (1 - shift_middle) * num_output_pixels

            temp_shift_row_top    = start_output_row - shift_pixels_up_left
            temp_shift_col_left   = start_output_col - shift_pixels_up_left
            temp_shift_row_bottom = start_output_row + shift_pixels_down_right
            temp_shift_col_right  = start_output_col + shift_pixels_down_right

            start_slice_top = 0
            start_slice_left = 0 
            end_slice_bottom = num_output_pixels
            end_slice_right  = num_output_pixels

            start_image_top    = int(np.max((temp_shift_row_top, 0)))
            start_image_left   = int(np.max((temp_shift_col_left, 0)))
            end_image_bottom   = int(np.min((temp_shift_row_bottom, output_height)))
            end_image_right    = int(np.min((temp_shift_col_right,  output_width)))

            if temp_shift_row_top < 0:
                start_slice_top = int(-temp_shift_row_top)

            if temp_shift_col_left < 0:
                start_slice_left = int(-temp_shift_col_left)

            if temp_shift_row_bottom > output_height:
                end_slice_bottom = int(num_output_pixels - (temp_shift_row_bottom - output_height))

            if temp_shift_col_right > output_width:
                end_slice_right  = int(num_output_pixels - (temp_shift_col_right - output_width))

            output_slice = np.zeros((num_output_pixels, num_output_pixels, 3))
            output_slice[start_slice_top: end_slice_bottom, start_slice_left: end_slice_right] = \
            slide[start_image_top: end_image_bottom, start_image_left: end_image_right]

            if i in all_paths_tissue_str_input:
                save_path = TISSUE_DIR_OUTPUT
            else:
                save_path = ALL_DIR_OUTPUT

            output_file_name = save_path + 'img_' + str(img_index) + '.jpg'

            try:
                cv2.imwrite(output_file_name, output_slice)
            except Exception as oerr:
                print('Error with saving:', oerr)


        except Exception as oerr:
            print('Error with slice:', oerr)


    data_root_tissue_output = pathlib.Path(TISSUE_DIR_OUTPUT)
    all_image_paths_tissue_output = list(data_root_tissue_output.glob('*'))
    all_paths_tissue_str_output   = [str(path) for path in all_image_paths_tissue_output]
    num_tissue_images_output = len(all_paths_tissue_str_output)

    data_root_all_output = pathlib.Path(ALL_DIR_OUTPUT)
    all_image_paths_all_output = list(data_root_all_output.glob('*'))
    all_paths_all_str_output   = [str(path) for path in all_image_paths_all_output]
    num_all_images_output = len(all_paths_all_str_output)

    if (num_tissue_images_output != num_tissue_images_input) or (num_all_images_output != num_all_images_input):
        print('ERROR: Number of output images not the same as number of input images')
        

#def test_part_1(testing_image_path_list, num_pixels=64, num_level=2):
def test_part_1(testing_image_path, num_pixels = 64, num_level=3):

#def testing(num_pixels, num_level):
    # for i in testing_image_path_list:
    #slide_path_test = i
    slide_path_test = testing_image_path
    tumor_mask_path_test = slide_path_test.split('.')[0]+'_mask.tif'
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

