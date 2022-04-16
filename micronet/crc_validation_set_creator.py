from PIL import Image


import numpy as np
import scipy as sp
import skimage.io
import os
import skimage.measure
import skimage.color
import random
from pathlib import Path
DIRECTORY_PATH = '/home/atlas/datasets/crc'
VALIDATION_FOLDER = 'val'
# perform reinhard color normalization
# Display results

import csv
from shutil import copyfile




if __name__ == '__main__':
    MSS_FRACTION= 0.2
    MSI_FRACTION = 0.2
    i = 0
    counter = 0
    for folder in ['test']:
        for imclass in ['MSS','MSIMUT']:
            localdir = DIRECTORY_PATH+'/'+folder+'/'+imclass+'/'
            destdir = DIRECTORY_PATH+'/'+VALIDATION_FOLDER+'/'+imclass+'/'
            for r, d, f in os.walk(localdir):
                for file in f:
                    try:
                        fraction =0
                        if imclass == 'MSS':
                            fraction = MSS_FRACTION
                        else:
                            fraction = MSI_FRACTION

                        if random.random() < fraction:
                            Path(localdir+file).rename(destdir+file)
                        counter += 1
                        if counter % 10000 == 0:
                            print(counter)
                    except Exception as ex:
                        print(localdir+file)
                        print(ex)
                        i = i+ 1
    print('number of corrupted files = ', i)


