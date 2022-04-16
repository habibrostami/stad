import histomicstk as htk
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization
import numpy as np
import scipy as sp
import skimage.io
import os
import skimage.measure
import skimage.color

DIRECTORY_PATH = '/home/deeplearning/datasets/stad/'
DEST_PATH = '/home/atlas/PycharmProjects/dataset/mecencostad/'

# perform reinhard color normalization
# Display results

import csv
from shutil import copyfile



TEST_FOLDER = 'test'
TRAIN_FOLDER = 'train'

if __name__ == '__main__':
    imReference = skimage.io.imread(DIRECTORY_PATH + 'train/MSS/' + 'blk-AAAVIASLIDNC-TCGA-D7-A6F0-01Z-00-DX1.png')[:,
                  :, :3]

    meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(imReference)
    print(meanRef, stdRef)


    for folder in ['train','test']:
        for imclass in ['MSS','MSIMUT']:
            localdir = DIRECTORY_PATH+'/'+folder+'/'+imclass+'/'
            destlocaldir = DEST_PATH+'/'+folder+'/'+imclass+'/'
            for r, d, f in os.walk(localdir):
                for file in f:
                    #print(f)
                    imInput = skimage.io.imread(
                        localdir+file)[:, :, :3]
                    #imNmzd = htk.preprocessing.color_normalization.reinhard(imInput, meanRef, stdRef)
                    imNmzd = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(imInput)
                    skimage.io.imsave(destlocaldir+file,imNmzd)
