from os import path, getcwd

DATASET_PATH = path.join(getcwd(), "d", "d_dataset")
TRAIN_DATASET_PATH = path.join(DATASET_PATH, 'train.hate.csv')
TEST_DATASET_PATH = path.join(DATASET_PATH, 'dev.hate.csv')

COMMENTS = 'comments'
LABEL = 'label'
STATUS = 'status'
MAX_LEN = 80
