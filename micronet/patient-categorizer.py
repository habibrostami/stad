from PIL import Image


import numpy as np
import scipy as sp
import skimage.io
import os
import skimage.measure
import skimage.color
import random
from pathlib import Path
DIRECTORY_PATH = '/home/atlas/datasets/stad'
DEST_DIRECTORY_PATH = '/home/atlas/datasets/patientstad'
FOLDER_NAME = 'test'
# perform reinhard color normalization
# Display results

import csv
from shutil import copyfile
import os
from shutil import copyfile


if __name__ == '__main__':
    dict = {}
    MSS_FRACTION= 0.2
    MSI_FRACTION = 0.2
    i = 0
    counter = 0
    for folder in ['train','test','val']:
        dict.clear()
        for imclass in ['MSS','MSIMUT']:
            localdir = DIRECTORY_PATH+'/'+folder+'/'+imclass+'/'
            destdir = DEST_DIRECTORY_PATH+'/'+folder+'/'+imclass+'/'
            for r, d, f in os.walk(localdir):
                for file in f:
                    tcga_name = file[17:]
                    pdir = tcga_name[:-4]
                    if tcga_name in dict:
                        dict[tcga_name] = dict[tcga_name] + 1
                        copyfile(localdir+file, destdir+pdir+'/'+file)
                    else:
                        i = i + 1
                        dict[tcga_name] = 1
                        os.mkdir(destdir+pdir)
                        copyfile(localdir+file, destdir+pdir+'/'+file)


    print(dict)
    print('number of patients = ', len(dict))
    print('i = ', i)



