import os

# local path
# DATASET_PATH = os.path.join(os.getcwd(), "d_dataset")

# google colab path
DATASET_PATH = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'd_dataset')
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train.hate.csv')
TEST_DATASET_PATH = os.path.join(DATASET_PATH, 'dev.hate.csv')
