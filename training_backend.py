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

def gen_image_paths(slide_path):
    img_num = slide_path.split('_')[1].strip(".tif")

    data_root_tumor = pathlib.Path('data/' + img_num + '/tumor')
    all_image_paths_tumor = list(data_root_tumor.glob('*'))
    num_tumor_images = len(all_image_paths_tumor)

    data_root_notumor = pathlib.Path('data/' + img_num + '/no_tumor')
    all_image_paths_notumor = list(data_root_notumor.glob('*'))
    random.shuffle(all_image_paths_notumor)
    all_image_paths_notumor = all_image_paths_notumor[0:num_tumor_images]

    all_image_paths = [str(path) for path in all_image_paths_tumor + all_image_paths_notumor]
    random.shuffle(all_image_paths)

    data_root = pathlib.Path('data/' + img_num)
    label_names = sorted(item.name for item in data_root.glob('*') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))

    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

    return all_image_paths, all_image_labels

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [128, 128])
    image /= 255.0  # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)

def create_tf_dataset(all_image_paths, all_image_labels):
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=8)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    BATCH_SIZE = 32

    steps_per_epoch = int(np.ceil(len(all_image_paths)/BATCH_SIZE))

    # Setting a shuffle buffer size larger than the dataset ensures that the data is completely shuffled.
    ds = image_label_ds.repeat()
    ds = ds.shuffle(buffer_size=4000)
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetches batches, asynchronously while the model is training.
    ds = ds.prefetch(1)


    return ds, steps_per_epoch


def training(num_pixels, num_level):
    # Load images

    slide_path = 'tumor_091.tif'
    tumor_mask_path = 'tumor_091_mask.tif'

    slide, tumor_mask  = load_image(slide_path, tumor_mask_path)

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

    ## Convert the mask from RGB to a black/white binary
    mask_image = mask_image[:,:,0]

    ## Determine the portions of the image that are tissue
    tissue_pixels = list(find_tissue_pixels(slide_image))

    ## Turn the tissue pixels into a mask
    tissue_regions = apply_mask(slide_image, tissue_pixels)

    ## Call the split function on the training data
    split_image_and_mask(slide_image, mask_image, tissue_regions, num_pixels, num_level, slide_path)

    ## Generate image paths and labels
    all_image_paths, all_image_labels = gen_image_paths(slide_path)

    ## Create tf.Dataset for training
    ds, steps_per_epoch = create_tf_dataset(all_image_paths, all_image_labels)

    ## Return training data
    return ds, steps_per_epoch

