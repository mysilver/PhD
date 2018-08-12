import os
import pickle
import queue
import random

import numpy as np
import tensorflow as tf
from ParaQuality.mlp import print_statistics
from ParaQuality.rnn import sentence_to_vec
from keras import Input, Model
from keras.layers import Dense, concatenate, Conv1D, MaxPool1D, Flatten, dot, Dropout, Bidirectional, LSTM
from sklearn.cross_validation import StratifiedKFold

from ParaQuality.old.evaluation.MSRP import msrp_dataset
from ParaQuality.old.features import OmniFeatureFunction
from utils.dataset import read_paraphrased_tsv_files
from utils.preprocess import remove_marks


def create_model(manual_features=31, max_length=25, filters=32, embedding_size=300, lstm_size=64, print_summary=False):
    manual_features = Input(shape=(manual_features,))
    input_1 = Input(shape=(max_length, embedding_size))
    input_2 = Input(shape=(max_length, embedding_size))

    conv = Conv1D(filters=filters, kernel_size=4)
    conv_out_1 = conv(input_1)
    conv_out_2 = conv(input_2)

    max_layer = MaxPool1D(pool_size=4)
    maxp_1 = max_layer(conv_out_1)
    maxp_2 = max_layer(conv_out_2)
    #
    flatter = Flatten()
    flat_1, flat_2 = flatter(maxp_1), flatter(maxp_2)
    dotcnn = dot([flat_1, flat_2], -1)

    shared_lstm = Bidirectional(LSTM(lstm_size, dropout=0.5, recurrent_dropout=0.3), merge_mode='concat', weights=None)
    encoded_a = shared_lstm(input_1)
    encoded_b = shared_lstm(input_2)
    dotrnn = dot([encoded_b, encoded_a], -1)

    drop = Dropout(rate=0.3)
    rnn = drop(concatenate([encoded_a, encoded_b, dotrnn], axis=-1))
    cnn = drop(concatenate([flat_1, flat_2, dotcnn], axis=-1))

    merged = concatenate([rnn, cnn, manual_features], axis=-1)
    dense1 = Dense(32, activation="softmax")(merged)

    predictions = Dense(1, name='output', activation="sigmoid")(dense1)
    # predictions = Activation("sigmoid")(dense)

    model = Model([manual_features, input_1, input_2], predictions)
    # model.compile(loss=reweight, optimizer='adam', metrics=['accuracy'])
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    if print_summary:
        model.summary()

    return model


def to_vector(dataset, max_length, save):
    # vectorize
    X, X1, X2, Y, T = [], [], [], [], []

    ffs = OmniFeatureFunction()
    for i, expression in enumerate(dataset):
        expression_vec = sentence_to_vec(expression, max_length)
        index = 0
        for instance in dataset[expression]:
            index += 1
            paraphrase = instance[0]
            paraphrase_vec = sentence_to_vec(paraphrase, max_length)
            features = ffs.extract(expression, paraphrase, index % 3)
            X.append(features)
            X1.append(expression_vec)
            X2.append(paraphrase_vec)
            Y.append(0 if instance[1] == 'invalid' else 1)
            T.append(expression + "<==>" + paraphrase)
        print("Dataset Processed", int((i + 1) / len(dataset) * 100), "%")
    if save:
        pickle.dump((X, X1, X2, Y, T), open(save, 'wb'))

    return np.array(X), np.array(X1), np.array(X2), np.array(Y), np.array(T)


def get_or_create_features_file(dataset_feature_path, datasets_path, max_length=25, shuffle=True):
    if os.path.isfile(dataset_feature_path):
        with open(dataset_feature_path, 'rb') as f:
            X, X1, X2, Y, T = pickle.load(f)
    else:
        if not msrp:
            dataset = read_paraphrased_tsv_files(datasets_path, processor=remove_marks)
        else:
            dataset = msrp_dataset(datasets_path, processor=remove_marks)
        X, X1, X2, Y, T = to_vector(dataset, max_length, save=dataset_feature_path)

    if shuffle:
        z = list(zip(X, X1, X2, Y, T))
        random.shuffle(z)
        X, X1, X2, Y, T = zip(*z)

    X = np.array(X)
    X1 = np.array(X1)
    X2 = np.array(X2)
    Y = np.array(Y)
    T = np.array(T)

    print("Input :", X1.shape, X2.shape)
    print("Output :", Y.shape)
    return X, X1, X2, Y, T


if __name__ == "__main__":

    batch_size = 128  # the more is the size, the less fn
    epochs = 75
    max_length = 25
    filters = 32
    lstm_size = 64
    embedding_size = 300
    callbacks = []
    msrp = False

    datasets = "../paraphrasing-data/crowdsourced"
    feature_file = "../paraphrasing-data/hybrid_feature_file.pickle"

    if msrp:
        feature_file = "../paraphrasing-data/hybrid_feature_file_msrq.pickle"
        datasets = "/media/may/Data/ParaphrsingDatasets/MRPC/original/msr-para-train.tsv"

    X, X1, X2, Y, T = get_or_create_features_file(feature_file, datasets, max_length, False)
    z = list(zip(X, X1, X2, Y, T))
    random.shuffle(z)
    X, X1, X2, Y, T = zip(*z)

    T = np.array(T)
    X1 = np.array(X1)
    X2 = np.array(X2)
    X = np.array(X)
    Y = np.array(Y)
    # X = np.reshape(X, (X.shape[0], X.shape[2]))
    print("Shaped after merging:", X.shape)
    with tf.device('/gpu:1'):
        kfold = StratifiedKFold(Y, n_folds=5)
        ttn, tfp, tfn, ttp = 0, 0, 0, 0
        for train, test in kfold:
            model = create_model(31, max_length, filters=filters, embedding_size=embedding_size, lstm_size=lstm_size)
            model.fit([X[train], X1[train], X2[train]], Y[train], batch_size=batch_size,
                      epochs=epochs, callbacks=callbacks,
                      verbose=0)
            scores = model.evaluate([X[test], X1[test], X2[test]], Y[test], verbose=1)
            y_pred_v = model.predict([X[test], X1[test], X2[test]])
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            y_pred = (y_pred_v >= 0.4)
            tn, fp, fn, tp = print_statistics(Y[test], y_pred)
            ttn += tn
            tfp += fp
            tfn += fn
            ttp += tp
            print("Threshold  > Acc: %.2f%%" % (100 * (tp + tn) / (tp + tn + fn + fp)))

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

        print("tn:", ttn, "fp:", tfp, "fn:", tfn, "tp:", ttp)
        print("Accuracy:", (ttp + ttn) / (ttp + ttn + tfp + tfn))
