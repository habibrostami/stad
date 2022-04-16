crc_checkpoint_directory = '/home/atlas/PycharmProjects/dqn/'
crc_checkpoint_filepath = crc_checkpoint_directory + 'crc_'+str(0)+ 'test_best_model.h5'
crc_rl_checkpoint_filepath = '/home/atlas/PycharmProjects/dqn/crc_rl_std_best_model.h5'
crc_tta_checkpoint_filepath = crc_checkpoint_directory + 'crc_'+str(0)+ 'tta_test_crc_best_model.h5'
crc_last_checkpoint_filepath=crc_checkpoint_directory+'crc_last_model.h5'
CRC_DIRECTORY_PATH = '/home/atlas/datasets/crc/'





checkpoint_directory = '/home/atlas/PycharmProjects/dqn/'
checkpoint_filepath = checkpoint_directory + 'super_epoch_'+str(0)+ 'test_std_best_model.h5'
rl_checkpoint_filepath = '/home/atlas/PycharmProjects/dqn/rl_std_best_model.h5'
tta_checkpoint_filepath = checkpoint_directory + 'super_epoch_'+str(0)+ 'tta_test_std_best_model.h5'
last_checkpoint_filepath=checkpoint_directory+'last_model.h5'
DIRECTORY_PATH = '/home/atlas/datasets/stad/'
DIRECTORY_PATH_DISTRIBUT_CHECK = '/home/atlas/datasets/stad/distribution-check/'

MODELS_FOLDER = '/home/atlas/PycharmProjects/dqn/savedmodels/'
RESULTS_XCELL = '/home/atlas/PycharmProjects/dqn/results.xlsx'

TRAIN_FOLDER = 'train'
TEST_FOLDER = 'test'
TEST_ORG_FOLDER = 'test-orig'
AUGMENTED_TRAIN_FOLDER='train_augmented'
AUGMENTED_VAL_FOLDER = 'val_augmented'
AUGMENTED_TEST_FOLDER = 'test_augmented'
VALIDATION_FOLDER = 'val'
MISSCLASSIFIED_FOLDER = 'miss'
TTA_DIRECTORY_PATH = '/home/atlas/datasets/stadnormtta'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELES = 3
MSIMUT_CLASS = 0
MSS_CLASS = 1
action_shape = 2
state_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELES)