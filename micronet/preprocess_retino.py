import csv
from shutil import copyfile


DIRECTORY_PATH = '/home/atlas/PycharmProjects/dataset/512x512-optimized/'
DEST_PATH = '/home/atlas/PycharmProjects/dataset/cleaned-512_512/'
TEST_FOLDER = 'test'
TRAIN_FOLDER = 'train'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'

if __name__ == '__main__':

    for folder in ['train','test']:
        with open(DIRECTORY_PATH + folder+'.csv', newline='') as f:
            reader = csv.reader(f)
            i = 0
            for row in reader:
                i = i + 1
                if i <= 1 :
                    continue

                if int(row[2]) >= 10 :
                    filename = DIRECTORY_PATH + folder +'/'+ row[1] + '/' + row[0]
                    dest =  DEST_PATH + folder +'/'+ row[1] + '/' + row[0]
                    copyfile(filename, dest)
                    print(row)