from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix,  classification_report

import models
import  gc
import tensorflow as tf
import numpy as np
import keras
from keras import Model,  optimizers
from keras import layers
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception

from keras.layers import Flatten, BatchNormalization, Dropout, Dense
from sklearn.utils import class_weight

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
IMAGE_CHANNELES = 3

epochs = 10
step_per_epoch = 100
validation_step = 1000
state_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELES)
action_shape = 2

DIRECTORY_PATH = '/home/atlas/PycharmProjects/dataset/twoclass-512_512/'
TRAIN_FOLDER = 'train'
TEST_FOLDER = 'test'
VALIDATION_FOLDER = 'val'

train_data_gen = ImageDataGenerator(rescale=1.0 / 255, ).flow_from_directory(DIRECTORY_PATH + TRAIN_FOLDER,
                                                                                  class_mode='categorical',
                                                                                  batch_size=1, shuffle=True,
                                                                                  target_size=(
                                                                                  IMAGE_HEIGHT, IMAGE_WIDTH)
                                                                                  )
test_data_gen = ImageDataGenerator(rescale=1.0 / 255, ).flow_from_directory(DIRECTORY_PATH + TEST_FOLDER,
                                                                                 class_mode='categorical', target_size=(
    IMAGE_HEIGHT, IMAGE_WIDTH)
                                                                                 )
#validation_data_gen = ImageDataGenerator(rescale=1.0 / 255, ).flow_from_directory(
#    DIRECTORY_PATH + VALIDATION_FOLDER,
#    class_mode='categorical', target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
#    )


classifier = models.get_InceptionRes(state_shape, action_shape, 0.0001, qlearning=False)


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(DIRECTORY_PATH + TRAIN_FOLDER,
                                                 target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                 batch_size=5,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(DIRECTORY_PATH + TEST_FOLDER,
                                            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                            batch_size=5,
                                            class_mode='categorical')

class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(training_set.classes),
    training_set.classes)

train_class_weights = dict(enumerate(class_weights))

history = classifier.fit_generator(training_set, class_weight= class_weights,
                                   steps_per_epoch=2000,
                                   epochs=20,
                                   validation_data=test_set,
                                   validation_steps=1300
                                   )

# y_pred = (y_pred > 0.5)


classifier.save("cnn.model")

import matplotlib.pyplot as plt


def append_history(history, h):
    '''
   This function appends the statistics over epochs
    '''
    try:
        history.history['loss'] = history.history['loss'] + h.history['loss']
        history.history['val_loss'] = history.history['val_loss'] + h.history['val_loss']
        history.history['acc'] = history.history['acc'] + h.history['acc']
        history.history['val_acc'] = history.history['val_acc'] + h.history['val_acc']
    except:
        history = h

    return history


def unfreeze_layer_onwards(model, layer_name):
    '''
        This layer unfreezes all layers beyond layer_name
    '''
    trainable = False
    for layer in model.layers:
        try:
            if layer.name == layer_name:
                trainable = True
            layer.trainable = trainable
        except:
            continue

    return model


def plot_performance(history):
    '''
	This function plots the train & test accuracy, loss plots
    '''

    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy v/s Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss v/s Epochs')
    plt.ylabel('M.S.E Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.tight_layout()
    plt.show()


plot_performance(history)

import h5py

classifier.save('my_full_model.h5')
classifier.save_weights('my_model_weights.h5', overwrite=True)
json_string = classifier.to_json()
with open('only_model.json', 'w') as f:
    f.write(json_string)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
