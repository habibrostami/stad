import shutil

from keras_preprocessing import image
from keras.preprocessing.image import  img_to_array
import PIL

from create_tta_data import HORIZONTAL_FLIP,NO_AUGMENT
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import keras
from keras.models import load_model

import models
import  gc

from settings import DIRECTORY_PATH, MISSCLASSIFIED_FOLDER, MSIMUT_CLASS, MSS_CLASS, \
    VALIDATION_FOLDER, rl_checkpoint_filepath, TEST_FOLDER,TRAIN_FOLDER, last_checkpoint_filepath
from keras.preprocessing.image import array_to_img, save_img
import os

STRATEGY_MAX_VALUE = 0
STRATEGY_AVERAGE_VALUE = 1
STRATEGY_CONFLICT_TO_BIAS = 2

NO_OF_FREEZED_LAYERS = 30
train_episodes = 150
MIN_REPLAY_SIZE = 2000
TRAIN_SAMPLE_SIZE = 2000
TRAIN_EPOCHS = 1
VALIDATION_STEPS = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.0001
RANDOM_SEED = 59
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BIAS_VALUE = 0.9

DEST_DIRECTORY_PATH = '/home/atlas/Desktop/for-hamed/'
#tf.random.set_seed(RANDOM_SEED)
from settings import rl_checkpoint_filepath
from settings import checkpoint_filepath, tta_checkpoint_filepath,DIRECTORY_PATH, MISSCLASSIFIED_FOLDER

env = None # gym.make('gym_stad:stad-v0')

#env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#print("Action Space: {}".format(env.action_space.shape))
#print("observation space: {}".format(env.observation_space.shape))


def find_label(model, file_path, label):
    img = image.load_img(file_path, target_size=(224, 224))
    # img.show()
    x = img_to_array(img) * 1.0 / 255  # rescacle the image
    x = x.reshape((1,) + x.shape)
    res = model.predict(x)
    # print('---------------res =', res)
    del img
    return res[0]



def find_and_copy_correct_for_explanation(path, folder_name):
    dict = {}
    best_model = load_model(path)
    best_model.summary()

    i = 0
    counter = 0
    for folder in [folder_name]:
        for imclass in ['MSS', 'MSIMUT']:
            localdir = DIRECTORY_PATH + '/' + folder + '/' + imclass + '/'
            destdir  = DEST_DIRECTORY_PATH + '/' + folder + '/' + imclass + '/'
            for r, d, f in os.walk(localdir):
                for file in f:
                    tcga_name = file[17:]
                    pdir = tcga_name[:-4]
                    if imclass == 'MSS':
                        label = 1
                    else:
                        label = 0
                    file_path = localdir + file
                    dest_file_path = destdir + file
                    res = find_label(best_model, file_path, label)
                    if tcga_name in dict:
                        if label == np.argmax(res):
                            dict[tcga_name][1].append(file_path)
                        else:
                            dict[tcga_name][2].append(file_path)
                    else:
                        if label == np.argmax(res):
                            shutil.copyfile(file_path,dest_file_path)

    return dict

if __name__ == '__main__':
    model_path = DEST_DIRECTORY_PATH + 'super_epoch_0test_std_best_model.h5'
    find_and_copy_correct_for_explanation(model_path,'test')