import numpy as np
import cv2

label_img = cv2.imread('CamVid/train/images/0001TP_006690.png')
label_img = label_img[:,:,0]
print(label_img)
print(label_img.shape)
print(np.unique(label_img))
print(np.ndim(label_img))
if np.ndim(label_img) == 3:
    print('here')
    label_img = np.squeeze(label_img, axis=-1)

print(label_img.shape)
print(label_img)
print(np.unique(label_img))
'''
num_classes = 32
heat_map = np.ones(shape=label_img.shape[0:2] + (num_classes,))
print(heat_map.shape)
print(heat_map)
print(np.unique(heat_map))

print('\n\n')
for i in range(num_classes):
    heat_map[:,:,i] = np.equal(label_img, i).astype('float32')

print(heat_map.shape)
print(heat_map)
'''
