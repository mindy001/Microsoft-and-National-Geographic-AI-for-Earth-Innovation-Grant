import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from partition_training_image_full_color import split_channels
import pandas as pd

def combined_images(nums,crop_type,bands,folders,final_nums):
    for num in nums:
        for type_ in crop_type:
            file_name = 'train/' + num + '_' 
            label_file_name = 'trainannot/' + num + '_' 
            #print(file_name)
            img_red = cv2.imread(file_name + bands[0] + '_' + type_ + '.png' )
            if(img_red is None):
                print('red ' + num)
                continue
            #print(img_red.shape)
            img_nir = cv2.imread(file_name + bands[1] + '_' + type_ + '.png' )
            if(img_nir is None):
                print('nir ' + num)
                continue
            img_ndvi = cv2.imread(file_name + bands[2] + '_' + type_ + '.png' )
            if(img_ndvi is None):
                print('ndvi ' + num)
                continue
            img_label = cv2.imread(label_file_name + type_ + '.png')
            if(img_label is None):
                print('label ' + num)
                continue
            #img_label = np.where(img_label==2,60,img_label)
            #img_label = np.where(img_label==1,255,img_label)
            
            combined = np.dstack((img_nir[:,:,0],img_red[:,:,0],img_ndvi[:,:,0]))
            #print(combined.shape)
            #plt.imshow(combined)
            #plt.show() 
            #print(folders[0] + '/' + num + '_combined_image.png')
            cv2.imwrite(folders[0] + '/' + num + '_combined_' + type_ + '_image.png',combined)
            cv2.imwrite(folders[1] + '/' + num + '_label_' + type_ + '_image.png', img_label)
            if(num not in final_nums):
                final_nums.append(num)
            #print('written')
            #break
        #break
    #break
    return final_nums

f = open('trainRed.txt')

bands = ['red','nir','ndvi']
crop_type = ['crop','weed']

nums = [line.split('/')[-1].split('_')[0] for line in f]
nums = np.unique(np.array(nums))
print(nums)
final_nums = []

folders = ['combined_full_images','train_full_label','train_partition_images', 'train_partition_ground_truth']

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

final_nums = combined_images(nums,crop_type,bands,folders,final_nums)

f.close()


file_num = file_label_num =  1
for num in final_nums:
    for type_ in crop_type:
        name = folders[0] + '/' + num + '_combined_' + type_ + '_image.png'
        label_name = folders[1] + '/' + num + '_label_' + type_ + '_image.png'
        file_num = split_channels(name,file_num,folders[2],True)
        file_label_num = split_channels(label_name,file_label_num,folders[3],False)
        

values = pd.Series(range(1,file_num))
values = values.to_frame()
values.columns=['name']
values.to_csv('train.csv',index=False)
