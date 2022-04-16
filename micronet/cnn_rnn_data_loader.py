import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.keras.datasets import cifar10
import albumentations as alb
from matplotlib import pyplot as plt
from cnn_rnn_settings import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELES
from cnn_rnn_settings import DIRECTORY_PATH, BATCH_SIZE, AUGMENT_SIZE

state_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELES)
from tensorflow.keras.utils import to_categorical

class MSIMSS:
    def __init__(self):
        self.train_steps = 90592 // BATCH_SIZE
        self.val_steps = 1977// BATCH_SIZE
        self.test_steps = 80882 // BATCH_SIZE

    def load_on_one_patioent(self,dir, batch_size=1):

        #dir = random.sample(list, 1)[0]
        if dir.find('MSIMUT') != -1:
            class_label = [1, 0]
        else:
            class_label = [0, 1]

        patient_list = []

        f = os.listdir(dir)
        #random.shuffle(f)
        tile_no = len(f)
        for k in range(tile_no//batch_size):
            files = f[k*batch_size:(k+1)*batch_size]
            for i, image_path in enumerate(files):
                image = tf.keras.preprocessing.image.load_img(dir+'/'+image_path)
                input_arr = tf.keras.preprocessing.image.img_to_array(image)/255.0
                patient_list.append(input_arr)

            input_arr = np.array(patient_list)
            tensors = tf.convert_to_tensor(input_arr)
            x_train = tf.ragged.stack(tensors)#.to_tensor()
            y_train = tf.convert_to_tensor([class_label])
            yield x_train, y_train


    def load_msi_mm_data_generator(self,folder, batch_size=1):
        list = []
        for imclass in ['MSS', 'MSIMUT']:
            localdir =  DIRECTORY_PATH + folder + '/' + imclass + '/'
            subfolders = [f.path for f in os.scandir(localdir) if f.is_dir()]
            list.extend(subfolders)

        while True:
            random.shuffle(list)
            for dir in list:
                #dir = random.sample(list, 1)[0]
                if dir.find('MSIMUT') != -1:
                    class_label = [1, 0]
                else:
                    class_label = [0, 1]

                patient_list = []

                f = os.listdir(dir)
                random.shuffle(f)
                sample_size = min(len(f), batch_size)
                files = random.sample(f, sample_size)

                for i, image_path in enumerate(files):
                    image = tf.keras.preprocessing.image.load_img(dir+'/'+image_path)
                    input_arr = tf.keras.preprocessing.image.img_to_array(image)/255.0
                    patient_list.append(input_arr)

                input_arr = np.array(patient_list)
                tensors = tf.convert_to_tensor(input_arr)
                x_train = tf.ragged.stack(tensors)#.to_tensor()
                y_train = tf.convert_to_tensor([class_label])
                yield x_train, y_train


class CIFAR10:
    def __init__ (self):
        self.aug = alb
        (x_train, y_train), (x_test,y_test) = cifar10.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        test_len = len(y_test)
        val_len = test_len // 5
        x_val = x_test[:val_len]
        y_val = y_test[:val_len]
        x_test = x_test[val_len:]
        y_test = y_test [val_len:]
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        self.y_val = to_categorical(y_val)
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.train_steps = len(y_train) // BATCH_SIZE
        self.val_steps = len(y_val) // BATCH_SIZE
        self.test_steps = len(y_test) // BATCH_SIZE

    def group_augment_data(self, data, agumentation_size, batch_size):
        if data == 'train':
            tensor = self.x_train
            label = self.y_train
        elif data == 'val':
            tensor = self.x_val
            label = self.y_val
        else:
            tensor = self.x_test
            label = self.y_test

        transform = self.aug.Compose([
            #self.aug.CLAHE(),
            self.aug.RandomRotate90(),
            self.aug.HorizontalFlip(0.5),
            self.aug.VerticalFlip (0.5),
            #self.aug.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
            #self.aug.Blur(blur_limit=3),
            self.aug.OpticalDistortion(),
        ])
        random.seed(42)
        sample_size = len(tensor)
        while True:
            for i in range(0,sample_size - batch_size,batch_size):
                res_x = []
                res_y = []
                for t in range(i,i+batch_size):
                    input_batch = np.zeros((agumentation_size, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELES))
                    for k in range(agumentation_size):
                        input_batch[k] = transform(image=tensor[t])['image']
                    np.random.shuffle(input_batch)
                    res_x.append(input_batch)
                    res_y.append(label[t])
                x_return = tf.convert_to_tensor(res_x)
                y_return = tf.convert_to_tensor(res_y)
                yield x_return,  y_return

    def get_data(self,data,  batch_size):
        if data == 'train':
            tensor = self.x_train
            label = self.y_train
        elif data == 'val':
            tensor = self.x_val
            label = self.y_val
        else:
            tensor = self.x_test
            label = self.y_test
        sample_size = len(tensor)
        while True:
            for i in range(0,sample_size - batch_size,batch_size):
                res_x = []
                res_y = []
                for t in range(i,i+batch_size):
                    res_x.append(tensor[t])
                    res_y.append(label[t])
                x_return = tf.convert_to_tensor(res_x)
                y_return = tf.convert_to_tensor(res_y)
                yield x_return,  y_return

if __name__ == "__main__":
    c10 = CIFAR10()
    x,y = next(c10.group_augment_data( 'train', AUGMENT_SIZE, BATCH_SIZE))
    print(y.shape)
