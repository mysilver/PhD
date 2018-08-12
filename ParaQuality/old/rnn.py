import os
import pickle
import queue
import random

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import LSTM, Dense, concatenate, Bidirectional
from sklearn.cross_validation import StratifiedKFold

from ParaQuality.old.mlp import print_statistics
from utils.dataset import read_paraphrased_tsv_files
from utils.preprocess import remove_marks, tokenize
from utils.service import word2vec_client as word2vec


def create_model(max_length, lstm_size, embedding_size=300, print_summary=False, compile=True):
    tweet_a = Input(shape=(max_length, embedding_size))
    tweet_b = Input(shape=(max_length, embedding_size))

    # embedding = Embedding(top_words + 2, embedding_size, input_length=max_length)
    #
    # embedding_tweet_a = embedding(tweet_a)
    # embedding_tweet_b = embedding(tweet_b)

    shared_lstm = Bidirectional(LSTM(lstm_size, dropout=0.4, recurrent_dropout=0.2), merge_mode='concat', weights=None)

    # encoded_a = shared_lstm(embedding_tweet_a)
    # encoded_b = shared_lstm(embedding_tweet_b)
    encoded_a = shared_lstm(tweet_a)
    encoded_b = shared_lstm(tweet_b)

    merged_vector = concatenate([encoded_a, encoded_b], axis=-1)
    predictions = Dense(1, activation='sigmoid')(merged_vector)

    if not compile:
        return tweet_a, tweet_b, predictions

    model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if print_summary:
        model.summary()

    return model


def sentence_to_vec(sentence, max_length):
    ret = np.zeros((max_length, 300))
    for i, t in enumerate(tokenize(sentence)):

        if i == max_length:
            break

        vec = word2vec.vector(t)
        if not vec:
            vec = word2vec.vector("unknown")

        ret[i] = vec

        if len(ret) == max_length - 1:
            break

    return ret


def to_vector(dataset, max_vec, save=None):
    X1, X2, Y, T = [], [], [], []

    # vectorize
    for i, expression in enumerate(dataset):
        expression_vec = sentence_to_vec(expression, max_vec)
        for instance in dataset[expression]:
            paraphrase = instance[0]
            features = sentence_to_vec(paraphrase, max_vec)
            X1.append(expression_vec)
            X2.append(features)
            Y.append(0 if instance[1] == 'invalid' else 1)
            T.append(expression + "<==>" + paraphrase)
        print("Dataset Processed", int((i + 1) / len(dataset) * 100), "%")
    if save:
        pickle.dump((X1, X2, Y, T), open(save, 'wb'))

    return np.array(X1), np.array(X2), np.array(Y), np.array(T)


def get_or_create_features_file(dataset_feature_path, datasets, max_vec, shuffle=True):
    # Read the file in the following format:
    # expression    paraphrase  [valid/invalid]
    if os.path.isfile(dataset_feature_path):
        with open(dataset_feature_path, 'rb') as f:
            X1, X2, Y, T = pickle.load(f)


    else:
        dataset = read_paraphrased_tsv_files(datasets, processor=remove_marks)
        X1, X2, Y, T = to_vector(dataset, max_vec, save=dataset_feature_path)

    if shuffle:
        z = list(zip(X1, X2, Y, T))
        random.shuffle(z)
        X1, X2, Y, T = zip(*z)

    X1 = np.array(X1)
    X2 = np.array(X2)
    Y = np.array(Y)
    T = np.array(T)

    print("Input :", X1.shape, X2.shape)
    print("Output :", Y.shape)
    return X1, X2, Y, T


if __name__ == "__main__":

    batch_size = 128
    epochs = 75
    max_length = 25
    lstm_size = 64
    callbacks = []
    dataset_feature_path = "../paraphrasing-data/rnn-features.pickle"
    datasets = "../paraphrasing-data/crowdsourced"

    X1, X2, Y, T = get_or_create_features_file(dataset_feature_path, datasets, max_length)

    with tf.device('/gpu:1'):
        kfold = StratifiedKFold(Y, n_folds=5)

        for train, test in kfold:
            model = create_model(max_length, lstm_size)
            model.fit([X1[train], X2[train]], Y[train], batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                      verbose=0)
            scores = model.evaluate([X1[test], X2[test]], Y[test], verbose=1)
            y_pred_v = model.predict([X1[test], X2[test]])
            y_pred = (y_pred_v >= 0.5)
            print_statistics(Y[test], y_pred)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

            y_pred = y_pred.tolist()
            y_test = Y[test]
            t_test = T[test]

            q = queue.PriorityQueue()
            for i in range(len(y_pred_v)):
                pred_class = y_pred[i]
                real_class = y_test[i] == 1

                y = y_pred_v[i][0]
                y_r = y_test[i]

                if pred_class[0] != real_class:
                    q.put((- abs(y - y_r), y_r, y, t_test[i]))

                if q.qsize() == 20:
                    break

            while not q.empty():
                score, real, stimated, text = q.get()
                # print(real, stimated, text)  # features
