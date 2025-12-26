import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from PIL import Image
import os
from os import listdir
from os.path import isfile, join

def split_channels(file_, file_num,folder,color):
    print(file_)
    try:
        image = Image.open(file_)
    except Exception:
        print('No file')
        return file_num
    print(image.size)
    print(image)
    array = asarray(image)
    print(array.shape)


    patch_size = 256

    #channels = ['train_red', 'train_green', 'train_blue']
    #channels = ['train_partition_images']
    #color = ['red', 'green', 'blue']
    #color = ['full_color']
    width_after = patch_size
    num = file_num
    #num = 1 
    print('New channels\n\n')
    for width in range(0, image.size[0], patch_size):
        print('width:')
        print(width)
        height_after = patch_size
        print('num:')
        print(num)
        print('height')
        for height in range(0, image.size[1], patch_size):
            print(height)
            patch = array[height:height_after,width:width_after, :]
            patch = Image.fromarray(patch)
            #patch.save(channels[0] + '/' + color[0] + '_training_' + str(num) + '.png')
            if(color):
                patch.save(folder + '/' + 'full_color' + '_training_' + str(num) + '.png')
            else:
                patch.save(folder + '/' + 'ground_truth' + '_training_' + str(num) + '.png')
            #plt.savefig('../training/train_gt/ground_truth_' + str(num) + '.tif')
            num+=1
            height_after+=patch_size
            if(height_after > image.size[1]):
                break

        width_after+=patch_size
        if(width_after > image.size[0]):
            break
    return num

'''
print('Why')
full_color_images_folder = 'full_color_images/'
files = [f for f in listdir(full_color_images_folder) if isfile(join(full_color_images_folder, f))]
file_num = 1
#folders = ['train_red', 'train_green', 'train_blue']
folders = ['train_full_color']
files.sort()
print(files)
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

for file_ in files:
    file_num = split_channels(full_color_images_folder + file_, file_num) 
'''
