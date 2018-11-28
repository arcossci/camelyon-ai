import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
from skimage.color import rgb2gray
import cv2
import os
import random
import pathlib
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
import pandas as pd


from training_backend import load_image, read_slide, find_tissue_pixels, apply_mask, load_and_preprocess_image, preprocess_image

def initialize_directories_test(slide_path):
    BASE_DIR = os.getcwd()

    img_num = slide_path.split('_')[1].strip(".tif")

    DATA = 'data/'
    IMG_NUM_FOLDER = img_num + '/'
    TISSUE_FOLDER = 'tissue_only/'
    ALL_FOLDER = 'all/'

    DATA_DIR = os.path.join(BASE_DIR, DATA)
    IMG_NUM_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER)
    TISSUE_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, TISSUE_FOLDER)
    ALL_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, ALL_FOLDER)

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(IMG_NUM_DIR):
        os.mkdir(IMG_NUM_DIR)
    if not os.path.exists(TISSUE_DIR):
        os.mkdir(TISSUE_DIR)
    if not os.path.exists(ALL_DIR):
        os.mkdir(ALL_DIR)

    return DATA + IMG_NUM_FOLDER + TISSUE_FOLDER, DATA + IMG_NUM_FOLDER + ALL_FOLDER


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

def gen_image_paths(slide_path):
    img_num = slide_path.split('_')[1].strip(".tif")
    img_test_folder = 'tissue_only'

    data_root = pathlib.Path('data/' + img_num + '/' + img_test_folder)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    return  all_image_paths


def create_tf_dataset(all_image_paths):
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=8)
    image_test_ds = tf.data.Dataset.zip((image_ds,))

    ## Dataset parameters
    BATCH_SIZE = 32

    steps_per_epoch = int(np.ceil(len(all_image_paths) / BATCH_SIZE))
    # Setting a shuffle buffer size larger than the dataset ensures that the data is completely shuffled.
    ds = image_test_ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetches batches, asynchronously while the model is training.
    ds = ds.prefetch(1)

    return ds, steps_per_epoch


def tumor_predict_mask(test, all_image_paths, depth, width):

    img_num = np.zeros(len(all_image_paths))
    for i in range(len(all_image_paths)):
        img_num[i] = int(all_image_paths[i].strip('.jpg').split('/')[-1].split('_')[-1])

    # depth, width = int(np.ceil(slide_image.shape[0] / pixel_num)), int(np.ceil(slide_image.shape[1] / pixel_num))

    predictions = np.zeros((depth, width))
    conf_threshold = 0.7

    for i in range(len(test)):
        y = int(img_num[i] // width)
        x = int(np.mod(img_num[i], width))
        predictions[y, x] = int(test[i][1] > conf_threshold)

    return predictions

def heatmap_evaluation(predictions, mask_image, tissue_regions):

    # we only need to evaluate on areas which are tissue.
    # correct non tumor prediction count would be higher if we get credit for
    # predicting gray areas aren't tumors

    # find out the correct amount to scale predictions to match image
    scale = int(mask_image.shape[0]/predictions.shape[0])

    # create scaled prediction matix
    predictions_scaled = np.kron(predictions, np.ones((scale, scale)))

    # reshape everything to a 1D vector for easy computation
    predictions_scaled = predictions_scaled.reshape(-1)
    mask_image = mask_image.reshape(-1)
    tissue_regions = tissue_regions.reshape(-1)

    # only include entries that have tissue
    predictions_scaled = predictions_scaled[tissue_regions == 1]
    mask_image = mask_image[tissue_regions == 1]

    # evaluate
    p = precision_score(mask_image, predictions_scaled)
    print('Precision:', p)
    r = recall_score(mask_image, predictions_scaled)
    print('Recall:', r)
    f = f1_score(mask_image, predictions_scaled)
    print('F1:', f)
    cm = confusion_matrix(mask_image, predictions_scaled)
    df_cm = pd.DataFrame(cm,columns = ['Predicted 0', 'Predicted 1'])
    df_cm.index = ['Reality 0', 'Reality 1']
    print('Confusion Matrix:')
    df_cm = pd.DataFrame(cm,columns = ['Predicted 0', 'Predicted 1'])
    df_cm.index = ['Reality 0', 'Reality 1']
    print(df_cm)
    df_cm_percent = df_cm
    df_cm_percent['Predicted 0'] = 100*df_cm_percent['Predicted 0']/len(mask_image)
    df_cm_percent['Predicted 1'] = 100*df_cm_percent['Predicted 1']/len(mask_image)
    print(df_cm_percent)

def testing(num_pixels, num_level):

    slide_path_test = 'tumor_110.tif'
    tumor_mask_path_test = 'tumor_110_mask.tif'

    slide, tumor_mask = load_image(slide_path_test, tumor_mask_path_test)

    ## Read training image at slide level 3
    slide_image = read_slide(slide,
                             x=0,
                             y=0,
                             level=num_level,
                             width=slide.level_dimensions[num_level][0],
                             height=slide.level_dimensions[num_level][1])

    mask_image = read_slide(tumor_mask,
                            x=0,
                            y=0,
                            level=num_level,
                            width=slide.level_dimensions[num_level][0],
                            height=slide.level_dimensions[num_level][1])

    ## Declare dimensions of slide image for use elsewhere
    depth, width = int(np.ceil(slide_image.shape[0] / num_pixels)), int(np.ceil(slide_image.shape[1] / num_pixels))

    ## Convert the mask from RGB to a black/white binary
    mask_image = mask_image[:, :, 0]

    ## Determine the portions of the image that are tissue
    
    tissue_pixels = list(find_tissue_pixels(slide_image))

    ## Turn the tissue pixels into a mask
    tissue_regions = apply_mask(slide_image, tissue_pixels)

    split_image_test(slide_image, tissue_regions, num_pixels, num_level, slide_path_test)

    ## Generate image paths and labels
    all_image_paths = gen_image_paths(slide_path_test)

    ## Create tf.Dataset for testing
    ds, steps_per_epoch = create_tf_dataset(all_image_paths)

    ## Return testing data
    return ds, slide_image, steps_per_epoch, all_image_paths, mask_image, tissue_regions, depth, width
