import cv2
import numpy as np
import time
import math
import multiprocessing as mp

import subprocess
import glob
import tarfile
from tqdm import tqdm
from os import path,makedirs
import os
import shutil
import pickle
from PIL import Image


image_folder='./val'

image_path = sorted(glob.glob(  image_folder + '/images/*'))
clip_length=2

for index in range(len(image_path)):
    beg = index - clip_length
    end = index + clip_length
    image_file = image_path[index].split('_')

    idx = int(image_file[3].strip('.png'))
    image = Image.open(image_path[index]).resize((228, 128))

    img_size = (228, 128)
    last_img_filename = image_path[index]
    flowfile=image_filename = '_'.join(image_file)
    flowpath='./val/flow/'+flowfile.split('/')[-1].split('.')[0]+'.npy'

    prev_image = image
    F=[]
    for i in range(idx - 2, idx - 2 * clip_length, -2):
        image_file[3] = str(i) + '.png'
        image_filename = '_'.join(image_file)

        try:
            image = Image.open(image_filename).resize(img_size, Image.NEAREST)
            last_img_filename = image_filename
        except Exception:
            #print(image_filename, last_img_filename)
            image = Image.open(last_img_filename).resize(img_size, Image.NEAREST)

        prvs = cv2.cvtColor(np.array(prev_image), cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        optical_flow = cv2.DualTVL1OpticalFlow_create()
        #optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = optical_flow.calc( curr,prvs, None)
        flow = np.array(flow).reshape((2,) + (128, 228))
        F.append(flow)

        prev_image = image
    Flow=np.array(F)
    last_img_filename
    #print(Flow, Flow.dtype)
    np.save(flowpath,Flow)


