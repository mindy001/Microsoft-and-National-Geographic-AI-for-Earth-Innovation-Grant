import cv2
import numpy as np

from keras.preprocessing.image import img_to_array



def data_gen_small(img_dir, lists, batch_size, dims):
    number_of_batches = np.ceil(len(lists) / batch_size)
    counter = 0
    #print('number of batches')
    #print(number_of_batches)
    while True:

        beg = batch_size * counter
        end = batch_size * (counter + 1)
        batch_files = lists[beg:end]
        #print('batch_files')
        #print(batch_files)

        ix = np.random.choice(np.arange(len(lists)), batch_size)
        imgs = []
        #print('ix')
        #print(ix)
        for i in batch_files.iloc():
            # images
            #print('Here')
            #img_path = img_dir + 'full_color_training_' + str(lists.iloc[i, 0]) + ".png"
            img_path = img_dir + 'full_color_training_' + str(i[0]) + ".png"
            #print(img_path)
            original_img = cv2.imread(img_path)[:, :, ::-1]
            #print(original_img.shape)
            #resized_img = cv2.resize(original_img, dims + [3])
            resized_img = cv2.resize(original_img, dims )
            array_img = img_to_array(resized_img) / 255
            #imgs.append(array_img)
            imgs.append(array_img)
            # masks
        counter+=1
        imgs = np.array(imgs)
        #print(imgs.shape)
        yield imgs

        if counter == number_of_batches:
            counter = 0
