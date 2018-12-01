# loading_ds_test.py

# In loading_ds_test.py we will need the following functions
# from training_backend.py
# •	preprocess_images
# •	load_and_preprocess_image

# And the following functions from testing_backend.py
# •	gen_image_paths
# •	create_tf_dataset
# •	tumor_predict_mask
# •	heatmap_evaluation

# This is basically the second part of the function testing
# in testing_backend.py

# This function will take in a single string [e.g. ‘img_003.tif’]
# and will output a single datastream for the test 
# image we want to use. It should also output the steps_per_epoch. 


import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
from skimage.color import rgb2gray
import cv2
import os
import random
import pathlib
import matplotlib.pyplot as plt

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [128, 128])
    image /= 255.0  # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)

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

    test = test[0:len(all_image_paths), :]
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


def test_part_2(training_image_path,model,tissue_regions, slide_image_test, mask_image, depth, width):
    ## Generate image paths and labels
    all_image_paths = gen_image_paths(slide_path_test)

    ## Create tf.Dataset for testing
    ds_test, steps_per_epoch_test = create_tf_dataset(all_image_paths)

    ## Predict on test data
    test_predicts = model.predict(ds_test, steps = steps_per_epoch_test)

    ## Create mask containing test predictions
    predictions = tumor_predict_mask(test_predicts, all_image_paths, depth, width)

    fig1, ax1 = plt.subplots()
	plt.imshow(slide_image_test)
	ax1.set_title("Original Image")

	fig2, ax2 = plt.subplots()
	plt.imshow(predictions)
	ax2.set_title("Predicted Tumor Mask")

	fig3, ax3 = plt.subplots()
	ax3.set_title("Actual Tumor Mask")
	plt.imshow(mask_image)

	heatmap_evaluation(predictions, mask_image, tissue_regions)



   