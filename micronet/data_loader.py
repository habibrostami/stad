
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
import numpy as np
from settings import DIRECTORY_PATH, TRAIN_FOLDER, VALIDATION_FOLDER, TEST_FOLDER, MSIMUT_CLASS, MSS_CLASS, MISSCLASSIFIED_FOLDER
from settings import IMAGE_HEIGHT, IMAGE_WIDTH

def get_image_batch_label(image):
    return np.argmax(image[1][0])


def get_image_batch_image(image):
    return image[0][0]


def get_data(folder):
    data_gen = ImageDataGenerator(rescale=1.0 / 255, ).flow_from_directory(DIRECTORY_PATH + folder,
                                                                                class_mode='categorical', batch_size=1,
                                                                                target_size=(
                                                                                    IMAGE_HEIGHT, IMAGE_WIDTH),
                                                                                shuffle=False,
                                                                                )
    print('class indices = ' ,data_gen.class_indices)
    # history = target_model.evaluate_generator(test_data_gen, steps=1300)
    # print(history)

    X_test = []
    Y_test = []
    print('----------------------------------------before loop test data collection ------------------')

    for batch in data_gen:
        print(batch)
        return 
    for i in range(int(data_gen.samples)):
        item = data_gen.next()
        X_test.append(get_image_batch_image(item))
        Y_test.append(get_image_batch_label(item))
    print('----------------------------------------after loop test data collection ------------------')
    return X_test, Y_test

get_data(TEST_FOLDER)