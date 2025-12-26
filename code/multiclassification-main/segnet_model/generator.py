import cv2
import numpy as np
import os

#from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array


def category_label(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            x[i, j, labels[i][j]] = 1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x


def data_gen_small(img_dir, mask_dir, lists, batch_size, dims, n_labels):
    while True:
        ix = np.random.choice(np.arange(len(lists)), batch_size)
        imgs = []
        labels = []
        for i in ix:
            # images
            #print('Here')
            img_path = img_dir + 'full_color_training_' + str(lists.iloc[i, 0]) + ".png"
            #print(img_path)
            #print(os.getcwd())
            #print(cv2.imread(img_path).shape)
            original_img = cv2.imread(img_path)[:, :, ::-1]
            #print(original_img.shape)
            #resized_img = cv2.resize(original_img, dims + [3])
            resized_img = cv2.resize(original_img, dims )
            array_img = img_to_array(resized_img) / 255
            imgs.append(array_img)
            # masks
            original_mask = cv2.imread(mask_dir + 'ground_truth_training_' + str(lists.iloc[i, 0]) + ".png")
            resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
            array_mask = category_label(resized_mask[:, :, 0], dims, n_labels)
            labels.append(array_mask)
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels
