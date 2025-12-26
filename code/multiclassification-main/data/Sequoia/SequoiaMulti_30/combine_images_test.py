import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from partition_testing_image_full_color import split_channels
import pandas as pd

def combined_images(nums,crop_type,bands,folders,final_nums):
    for num in nums:
        for type_ in crop_type:
            file_name = 'test/' + num + '_' 
            label_file_name = 'testannot/' + num

            #print(file_name + bands[0] + '_' + type_ + '.png')
            #print('test/0000_red_crop.png')
            img_red = cv2.imread(file_name + bands[0] + '_' + type_ + '.png' )
            if(img_red is None):
                print('red ' + num)
                continue
            #print(file_name + bands[0] + '_' + type_)
            #print(img_red.shape)
            img_nir = cv2.imread(file_name + bands[1] + '_' + type_ + '.png' )
            if(img_nir is None):
                print('nir ' + num)
                continue
            #print(file_name + bands[1] + '_' + type_)
            #print(img_nir.shape)
            img_ndvi = cv2.imread(file_name + bands[2] + '_' + type_ + '.png' )
            if(img_ndvi is None):
                print('ndvi ' + num)
                continue
            #print(file_name + bands[2] + '_' + type_)
            #print(img_ndvi.shape)
            img_label = cv2.imread(label_file_name + '.png')
            if(img_label is None):
                print('label ' + num)
                continue
            #img_label = np.where(img_label==2,60,img_label)
            #img_label = np.where(img_label==1,255,img_label)
            
            height = img_label.shape[0]
            width  = img_label.shape[1]
            
            img_red = cv2.resize(img_red, (width, height))
            img_nir = cv2.resize(img_nir, (width, height))
            img_ndvi= cv2.resize(img_ndvi,(width, height))
            '''
            try:
                combined = np.dstack((img_nir[:,:,0],img_red[:,:,0],img_ndvi[:,:,0]))
            except Exception:
                print('Wrong dimension')
                continue
            '''
            combined = np.dstack((img_nir[:,:,0],img_red[:,:,0],img_ndvi[:,:,0]))
            #print(combined.shape)
            #plt.imshow(combined)
            #plt.show() 
            #print(folders[0] + '/' + num + '_combined_image.png')
            cv2.imwrite(folders[0] + '/' + num + '_combined_test_' + type_ + '_image.png',combined)
            cv2.imwrite(folders[1] + '/' + num + '_label_test_' + type_ + '_image.png', img_label)
            if(num not in final_nums):
                final_nums.append(num)
            #print('written')
            #break
        #break
    #break
    return final_nums




f = open('testRed.txt')

bands = ['red','nir','ndvi']
crop_type = ['crop','weed']

nums = [line.split('/')[-1].split('_')[0] for line in f]
nums = np.unique(np.array(nums))
print(nums)
final_nums = []

folders = ['combined_full_test_images','test_full_label','test_partition_images', 'test_partition_ground_truth','test_partition_ground_truth_visible']

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

final_nums = combined_images(nums,crop_type,bands,folders,final_nums)

f.close()


file_num = file_label_num = file_label_num_visible = 1
for num in final_nums:
    for type_ in crop_type:
        name = folders[0] + '/' + num + '_combined_test_' + type_ + '_image.png'
        label_name = folders[1] + '/' + num + '_label_test_' + type_ + '_image.png'
        file_num = split_channels(name,file_num,folders[2],True,False)
        file_label_num = split_channels(label_name,file_label_num,folders[3],False,False)
        file_label_num_visible = split_channels(label_name,file_label_num_visible,folders[4],False,True)
        

values = pd.Series(range(1,file_num))
values = values.to_frame()
values.columns=['name']
values.to_csv('test.csv',index=False)

