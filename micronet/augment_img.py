from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


import Augmentor
import os

from settings import DIRECTORY_PATH, TRAIN_FOLDER, AUGMENTED_VAL_FOLDER, AUGMENTED_TRAIN_FOLDER
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


dir = DIRECTORY_PATH + TRAIN_FOLDER + '/'
target_dir = DIRECTORY_PATH + AUGMENTED_TRAIN_FOLDER + '/'

makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]
print(folders)
range_no = 4
for i in range(len(folders)):
    range_no -= 1 # 3 iterations for MSIMUT and 2 iterations for MSS

    fd = folders[i]
    tfd = target_folders[i]
    # rotation
    #p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    #p.gaussian_distortion(probability=1.0, grid_width=10,grid_height=10, magnitude= 3, corner = 'bell', method ='in')
    #p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    #p.histogram_equalisation(probability=1.0)
    #p.flip_left_right(probability=0.5)
    #for i in range(0):
    #  ""  p.process()
    #del p


    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.random_brightness(probability=0.8,min_factor=0.5,max_factor=1.5)
    #p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    #p.histogram_equalisation(probability=1.0)
    p.flip_left_right(probability=0.5)
    del p


    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    #p.random_brightness(probability=0.4,min_factor=0.8,max_factor=1.2)
    p.random_color(probability=0.8,min_factor=0.5,max_factor=1.5)
    #p.histogram_equalisation(probability=1.0)
    p.flip_left_right(probability=0.5)
    for i in range(range_no):
        p.process()
    del p

    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    #p.random_brightness(probability=0.4,min_factor=0.8,max_factor=1.2)
    p.random_contrast(probability=0.8,min_factor=0.5,max_factor=1.5)
    #p.histogram_equalisation(probability=1.0)
    p.flip_left_right(probability=0.5)
    for i in range(range_no):
        p.process()
    del p

    # skew
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.skew(probability=0.5, magnitude=0.2)  # max 45 degrees
    p.flip_left_right(probability=0.5)
    for i in range(range_no):
        p.process()
    del p


    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    #p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    #p.histogram_equalisation(probability=1.0)
    p.flip_left_right(probability=0.5)
    p.crop_centre(probability=1.0,percentage_area=0.5)
    p.resize(probability=1.0,width=224,height=244)
    for i in range(range_no):
        p.process()
    del p


    # random_distortion
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.random_distortion(probability=.5, grid_width=10, grid_height=10, magnitude=5)
    p.flip_left_right(probability=0.5)
    for i in range(range_no):
        p.process()
    del p

