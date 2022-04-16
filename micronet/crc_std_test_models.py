from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#from keras.engine.saving import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input

import models
import numpy as np
from sklearn.utils import class_weight
import keras
import tensorflow as tf
from keras.models import save_model
from augmentation_util import color_augment_patches, to_three_channel_gray_scale

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELES = 3

FREEZ_RATE = 0
NUMBER_OF_SUPER_EPOCHS = 5
EPOCHS = 30
state_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELES)
action_shape = 2
BATCH_SIZE = 16
NO_OF_FREEZED_LAYERS = 0
LEARNING_RATE = 0.000001

from settings import CRC_DIRECTORY_PATH, TRAIN_FOLDER, TEST_FOLDER, VALIDATION_FOLDER


from settings import crc_checkpoint_filepath,crc_last_checkpoint_filepath
from settings import crc_checkpoint_directory

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=crc_checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

lr_change_callback = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1,
    mode="auto",
    min_delta=0.01,
    cooldown=0,
    min_lr=0,
)




#validation_data_gen = ImageDataGenerator(rescale=1.0 / 255, ).flow_from_directory(
#    DIRECTORY_PATH + VALIDATION_FOLDER,
#    class_mode='categorical', target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
#    )


#33with tf.device("/gpu:0"):
#   print("tf.keras code in this scope will run on GPU")


train_datagen = ImageDataGenerator(
                                   rescale=1. / 255,
                                   #zoom_range=[1.0,1.1],
                                   #horizontal_flip=True,
                                   #vertical_flip=True,
                                   #brightness_range= [0.6,1.4],
                                   #preprocessing_function=color_augment_patches
                                   )

test_datagen = ImageDataGenerator(
                                rescale=1. / 255,

                                )
validation_datagen = ImageDataGenerator(
                                rescale=1. / 255,
                                #preprocessing_function=to_three_channel_gray_scale

)

training_set = train_datagen.flow_from_directory(CRC_DIRECTORY_PATH + TRAIN_FOLDER,
                                                 target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical', shuffle=True)

test_set = test_datagen.flow_from_directory(CRC_DIRECTORY_PATH + TEST_FOLDER,
                                            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical', shuffle=False)

validation_set = validation_datagen.flow_from_directory(CRC_DIRECTORY_PATH + VALIDATION_FOLDER,
                                            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical', shuffle=False)

class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(validation_set.classes),
    validation_set.classes)

val_class_weights = dict(enumerate(class_weights))



train_sample_size = training_set.samples
test_sample_size = test_set.samples
validation_sample_size = validation_set.samples

print('------------------------train samples  = ', train_sample_size)
def get_refresh_classifier():

    classifier = models.get_Xception(state_shape, action_shape, learning_rate=LEARNING_RATE, qlearning=False, no_freez_layers = NO_OF_FREEZED_LAYERS)
    return classifier


classifier = get_refresh_classifier()
print('weights  = ', val_class_weights)
history = classifier.fit(training_set, #class_weight= val_class_weights,
                                   steps_per_epoch=int(train_sample_size / BATCH_SIZE),
                                   epochs=EPOCHS,
                                   validation_data=validation_set,
                                   verbose=1,
                                   validation_steps=int(validation_sample_size/ BATCH_SIZE),
                                   callbacks=[model_checkpoint_callback ]#,lr_change_callback]
                                   )

save_model (classifier,crc_last_checkpoint_filepath)

def run_super_epoch(model,  super_epoch_no):
    checkpoint_filepath = crc_checkpoint_directory + 'super_epoch_'+str(super_epoch_no)+ 'std_best_model.h5'
    model_checkpoint_callback_super_epoch = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_acc',
        mode='max',
        save_best_only=True)
    history = model.fit_generator(training_set,  # class_weight= class_weights,
                                       steps_per_epoch=int(train_sample_size / BATCH_SIZE),
                                       epochs=EPOCHS, shuffle=True,
                                       validation_data=test_set, verbose=2,
                                       validation_steps=int(test_sample_size / BATCH_SIZE),
                                       callbacks=[model_checkpoint_callback_super_epoch,lr_change_callback]
                                       )


    return history, checkpoint_filepath

#learning_rate = 0.0001
#check_file = checkpoint_filepath
#for i in range(1,NUMBER_OF_SUPER_EPOCHS):
#    temp_model = load_model(check_file)
#    classifier = get_refresh_classifier()
#    classifier.set_weights(temp_model.get_weights())
#    classifier = models.freez_layers(classifier,i*FREEZ_RATE)
#    learning_rate = learning_rate / float(i)
#    classifier.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
#    history, check_file = run_super_epoch(classifier,i)



import matplotlib.pyplot as plt


def append_history(history, h):
    '''
   This function appends the statistics over epochs
    '''
    try:
        history.history['loss'] = history.history['loss'] + h.history['loss']
        history.history['val_loss'] = history.history['val_loss'] + h.history['val_loss']
        history.history['accuracy'] = history.history['accuracy'] + h.history['accuracy']
        history.history['val_accuracy'] = history.history['val_accuracy'] + h.history['val_accuracy']
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
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
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

#classifier.save('my_full_model.h5')
#classifier.save_weights('my_model_weights.h5', overwrite=True)
json_string = classifier.to_json()
with open('only_model.json', 'w') as f:
    f.write(json_string)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
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
