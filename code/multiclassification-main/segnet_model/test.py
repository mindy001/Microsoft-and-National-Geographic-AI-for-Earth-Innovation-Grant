import pandas as pd
from generator_test import data_gen_small
from model import segnet
import numpy as np
import os
import cv2
from plot_images import visualize
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array


def main():
    #test_list = pd.read_csv('gdrive/My Drive/Research/TTLAB/AI_For_Earth/Problem_2/test_2/test_2.csv')
    test_list = pd.read_csv('gdrive/My Drive/Research/TTLAB/AI_For_Earth/Problem_2/test_2/test.csv')
    testing_img_dir = 'gdrive/My Drive/Research/TTLAB/AI_For_Earth/Problem_2/multiclassification/data/SequoiaMulti_30/test_partition_images/'
    weights_path = 'gdrive/My Drive/Research/TTLAB/AI_For_Earth/Problem_2/test_2/650.hdf5'
    PRED_FOLDER = 'gdrive/My Drive/Research/TTLAB/AI_For_Earth/Problem_2/test_2/'
    img_path = 'gdrive/My Drive/Research/TTLAB/AI_For_Earth/Problem_2/multiclassification/data/SequoiaMulti_30/test_partition_images/full_color_training_1.png'

    batch_size = 10
    input_shape = (256,256,3)
    print('test_list')
    print(test_list)

    test_gen = data_gen_small(
        testing_img_dir,
        test_list,
        batch_size,
        #[args.input_shape[0], args.input_shape[1]],
        (input_shape[0], input_shape[1]),
    )

    n_labels = 3
    kernel = 3
    pool_size = (2,2)
    output_mode = 'softmax'
    model = segnet(
        input_shape, n_labels, kernel, pool_size, output_mode
    )

    
    model.load_weights(weights_path)

    '''
    original_img = cv2.imread(img_path)[:, :, ::-1]
    #print(original_img.shape)
    #resized_img = cv2.resize(original_img, dims + [3])
    resized_img = cv2.resize(original_img, (256,256) )
    array_img = img_to_array(resized_img) / 255

    output = model.predict(np.array([array_img]))

    image = np.reshape(output, (256,256,3))
    p = np.argmax(image, axis=-1)
    p = np.expand_dims(p, axis=-1)
    p = p * (255/3)
    p = p.astype(np.int32)
    print(p.shape)
    p = np.concatenate([p, p, p], axis=2)
    '''

    '''
    image_2 = visualize(np.argmax(image,axis=-1).reshape((256,256)), False)
    print(image_2.shape)
    print(image_id)
    print(np.unique(image_2))
    '''

    '''
    pred_dir = '4_epochs_predictions'
    name = 'img_1' + '.png'
    cv2.imwrite(os.path.join(PRED_FOLDER, pred_dir, name), p)
    '''


    imgs_mask_test = model.predict_generator(
        test_gen,
        steps=np.ceil(len(test_list) / batch_size)-1)


    '''
    #original_img = cv2.imread(img_path)[:, :, ::-1]
    original_img = np.array([cv2.imread(img_path)])
    #nop = np.array([None])
    #original_img = np.append(nop, original_img)
    print(original_img.shape)
    output = model.predict(original_img)
    pred = visualize(np.argmax(output[0],axis=1).reshape((256,256)), False)
    print(pred.shape)
    #plt.imshow(pred)
    #plt.show()
    cv2.imwrite(os.path.join('gdrive/My Drive/Research/TTLAB/AI_For_Earth/Problem_2/test_2/img_2_300_epoch.png'), output)
    #cv2.imwrite(os.path.join('gdrive/My Drive/Research/TTLAB/AI_For_Earth/Problem_2/test_2/img_50_epoch.png'), pred)
  
    '''

    pred_dir = '650_epochs_predictions'
    if not os.path.exists(os.path.join(PRED_FOLDER, pred_dir)):
        os.mkdir(os.path.join(PRED_FOLDER, pred_dir))


    for image, image_id in zip(imgs_mask_test, test_list.iloc()):
        #image = (image[:, :, 0]).astype(np.float32)
        #print(image_id)
        #print(image.shape)
        #print(np.unique(image[:,0] == image[:,1]))
        image = np.reshape(image, (256,256,3))
        #print('Image shape')
        #print(image.shape)
        #print(np.unique(image))
        #print(np.argmax(image,axis=-1))
        #print(np.unique((np.argmax(image,axis=-1))))
        #print('Here')
        p = np.argmax(image, axis=-1)
        p = np.expand_dims(p, axis=-1)
        p = p * (255/3)
        p = p.astype(np.int32)
        #print(p.shape)
        p = np.concatenate([p, p, p], axis=2)
        #image_2 = visualize(np.argmax(image,axis=-1).reshape((256,256)), False)
        #print(image_2.shape)
        #print(image_id)
        #print(np.unique(image_2))
        #print(image_id[0])
        if(image_id[0]%50 == 0):
          print(image_id)
        name = str(image_id[0]) + '.png'
        cv2.imwrite(os.path.join(PRED_FOLDER, pred_dir, name), p)
      
    print("Saving predicted cloud masks on disk... \n")
if __name__ == "__main__":
    main()

