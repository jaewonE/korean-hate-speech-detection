import os
import tensorflow as tf
from warnings import simplefilter
import sklearn.model_selection as sk

from paths import *
from constant import *
from preprocessing import get_preprocessing_data, get_unpreprocessing_data

simplefilter(action='ignore', category=FutureWarning)


def get_model(mode, len_word):
    dimension = 64
    regularizer5 = tf.keras.regularizers.l2(0.005)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len_word + 1, dimension,
                                  input_length=MAX_LEN, embeddings_regularizer=regularizer5),
        tf.keras.layers.LSTM(dimension, return_sequences=True,
                             kernel_regularizer=regularizer5, bias_regularizer=regularizer5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=regularizer5, bias_regularizer=regularizer5),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(
            8, activation='relu', kernel_regularizer=regularizer5, bias_regularizer=regularizer5),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model, os.path.join(os.getcwd(), 'drive', 'MyDrive', 'd_logs', mode)


def simulation(mode, epochs, usePreprocess_data):
    x_data, y_data, len_train_word = get_preprocessing_data(
        TRAIN_DATASET_PATH) if usePreprocess_data else get_unpreprocessing_data(TRAIN_DATASET_PATH)
    x_train, x_valid, y_train, y_valid = sk.train_test_split(
        x_data, y_data, test_size=0.2, shuffle=True, stratify=y_data, random_state=34)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

    x_test, y_test, len_test_word = get_preprocessing_data(TEST_DATASET_PATH)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    model, base_log_dir = get_model(mode, len_train_word)
    # model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=base_log_dir)
    model.fit(train_ds.shuffle(5000).batch(1024), epochs=epochs,
              validation_data=val_ds.batch(1024), verbose=2, callbacks=[tensorboard])
    model.save(os.path.join(os.getcwd(), 'drive', 'MyDrive', 'd_model', mode))
    score = model.evaluate(x_test, y_test)
    print(score)


simulation('base1000', 1000, True)
simulation('base100', 100, True)
simulation('not_preprocessing', 1000, False)
