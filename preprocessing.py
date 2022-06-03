from paths import *
from constant import *
import spell_checker

import tensorflow as tf
from pandas import read_csv
from itertools import groupby
from warnings import simplefilter
import numpy as np
simplefilter(action='ignore', category=FutureWarning)


pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    1000, char_level=True, oov_token="<OOV>")


# def my_one_hot_encoding(data):
#     u = ''.join(data)
#     u = list(set(u))
#     u.sort()
#     a_list = []
#     for i, d in enumerate(u):
#         a_list.append([i, d])
#     return a_list


# def one_hot_encoding(data_list):
#     tokenizer.fit_on_texts(data_list)
#     word_index = tokenizer.word_index
#     return word_index

def get_data_list(path):
    try:
        data = read_csv(path)
    except:
        print("Error: File not found")
        exit()
    return data


def data_pretreatment(data):
    # 중복된 comments 제거
    data.drop_duplicates(subset=[COMMENTS], inplace=True)

    # label의 값이 hate, offensive이면 1, none이면 0으로 하는 status 열 생성
    data[STATUS] = np.where(data[LABEL] == 'none', 0, 1)

    # 한글과 숫자, 띄워쓰기가 아닌 글자들은 제거
    data[COMMENTS] = data[COMMENTS].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]', '')

    # 필요없이 반복되는 문자열(예: ㅋㅋㅋㅋㅋ)을 줄이기
    data[COMMENTS] = data[COMMENTS].map(
        lambda x: ''.join(ch for ch, _ in groupby(x)))

    # 네이버 맞춤법 검사
    data[COMMENTS] = data[COMMENTS].map(
        lambda x: spell_checker.check(x).checked)
    return data


def get_preprocessing_data(csv_path):
    data = get_data_list(csv_path)
    data = data_pretreatment(data)

    data = data.dropna()
    data = data.mask(data.eq('None')).dropna()

    comment_list = data[COMMENTS].tolist()
    tokenizer.fit_on_texts(comment_list)
    train_seq = tokenizer.texts_to_sequences(comment_list)
    x_data = pad_sequences(train_seq, MAX_LEN)
    y_data = data[STATUS].tolist()
    return np.array(x_data), np.array(y_data), len(tokenizer.word_index)


def get_unpreprocessing_data(csv_path):
    data = get_data_list(csv_path)
    data[STATUS] = np.where(data[LABEL] == 'none', 0, 1)
    comment_list = data[COMMENTS].tolist()
    tokenizer.fit_on_texts(comment_list)
    train_seq = tokenizer.texts_to_sequences(comment_list)
    x_data = pad_sequences(train_seq, MAX_LEN)
    y_data = data[STATUS].tolist()
    return np.array(x_data), np.array(y_data), len(tokenizer.word_index)
