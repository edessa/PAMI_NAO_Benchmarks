import cv2
import numpy as np
import scipy.misc as misc

dataset = './Dataset/images/'

def get_visuals(img, np_arr):
    time_map = np.zeros((img.shape))
    for row in range(len(time_map)):
        for col in range(len(time_map[0])):
            if np_arr[row][col] == 1:
                time_map[row][col] = (255, 0, 0)
            elif np_arr[row][col] != -1:
                time_map[row][col] = (255, 0, 0)
    return time_map

for i in range(100000):
    try:
        print(i)
        img = cv2.imread(dataset + str(i) + '.png')
        np_arr = np.load(dataset + 'np_mask_' + str(i) + '.npy')

        time_map = get_visuals(img, np_arr)
        concat_img = np.concatenate((img, time_map), axis=1)
        cv2.imshow('Images', img)
        while cv2.waitKey(0) != ord('q'):
            pass
        cv2.destroyAllWindows()
    except FileNotFoundError as e:
        continue
