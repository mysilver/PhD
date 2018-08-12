import os
import pickle
import queue
import random

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPool1D, dot, concatenate, Activation, \
    average
from sklearn.cross_validation import StratifiedKFold

from ParaQuality.old.mlp import print_statistics, reweight
from utils.dataset import read_paraphrased_tsv_files
from utils.preprocess import tokenize, remove_marks


def sentence_to_vec(sentence, max_length):
    from ParaVec import word2vec
    ret = np.zeros((max_length, 300))
    for i, t in enumerate(tokenize(sentence)):
        vec = word2vec.vector(t)
        if vec is None:
            vec = word2vec.vector("unknown")

        ret[i] = vec

        if len(ret) == max_length - 1:
            break

    return ret


def find_diverse_seeds(expression, samples, seed_size=3):
    seeds = [expression]
    random.shuffle(samples)
    for s in samples:

        if len(seeds) == seed_size:
            break

        if s[1] == 'valid':
            seeds.append(s[0])

    return seeds, [s for s in samples if s[0] not in seeds]


def to_vector(dataset, max_vec, save=None):
    X1, X2, X3, X4, Y, T = [], [], [], [], [], []

    for i, expression in enumerate(dataset):
        # expression_vec = sentence_to_vec(expression, max_vec)
        seeds, samples = find_diverse_seeds(expression, dataset[expression])

        X2_c = sentence_to_vec(seeds[0], max_vec)
        X3_c = sentence_to_vec(seeds[1], max_vec)
        X4_c = sentence_to_vec(seeds[2], max_vec)

        for instance in samples:
            X1.append(sentence_to_vec(instance[0], max_vec))
            X2.append(X2_c)
            X3.append(X3_c)
            X4.append(X4_c)

            Y.append(0 if instance[1] == 'invalid' else 1)
            T.append(expression + "<==>" + instance[0])

        print("Dataset Processed", int((i + 1) / len(dataset) * 100), "%")
    if save:
        pickle.dump((X1, X2, X3, X4, Y, T), open(save, 'wb'))

    return np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(Y), np.array(T)


def get_or_create_features_file(dataset_feature_path, datasets, max_vec, shuffle=True):
    # Read the file in the following format:
    # expression    paraphrase  [valid/invalid]
    if os.path.isfile(dataset_feature_path):
        with open(dataset_feature_path, 'rb') as f:
            X1, X2, X3, X4, Y, T = pickle.load(f)

    else:
        dataset = read_paraphrased_tsv_files(datasets, processor=remove_marks)
        X1, X2, X3, X4, Y, T = to_vector(dataset, max_vec, save=dataset_feature_path)

    if shuffle:
        z = list(zip(X1, X2, X3, X4, Y, T))
        random.shuffle(z)
        X1, X2, X3, X4, Y, T = zip(*z)

    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)
    Y = np.array(Y)
    T = np.array(T)
    print("Input :", X1.shape)
    print("Output :", Y.shape)
    return X1, X2, X3, X4, Y, T


def create_model(shape, filters, print_summary=False, compile=True):
    input_p = Input(shape=shape)
    input_s1 = Input(shape=shape)
    input_s2 = Input(shape=shape)
    input_s3 = Input(shape=shape)

    # Layer 1: Convolution
    conv = Conv1D(filters=filters, kernel_size=4)
    conv1_p = conv(input_p)

    conv1_s1 = conv(input_s1)
    conv1_s2 = conv(input_s2)
    conv1_s3 = conv(input_s3)

    # Layer 2: MaxPooling
    # maxp1_p = conv1_p
    # maxp1_s1 = conv1_s1
    # maxp1_s2 = conv1_s2
    # maxp1_s3 = conv1_s3
    maxp1_p = MaxPool1D(pool_size=6)(conv1_p)
    maxp1_s1 = MaxPool1D(pool_size=6)(conv1_s1)
    maxp1_s2 = MaxPool1D(pool_size=6)(conv1_s2)
    maxp1_s3 = MaxPool1D(pool_size=6)(conv1_s3)


    # Layer 3: Convolution
    conv2 = Conv1D(filters=int(filters / 2), kernel_size=2)
    conv2_p = conv2(maxp1_p)
    conv2_s1 = conv2(maxp1_s1)
    conv2_s2 = conv2(maxp1_s2)
    conv2_s3 = conv2(maxp1_s3)

    # Layer 4: MaxPooling
    maxp2_p = MaxPool1D(pool_size=2)(conv2_p)
    maxp2_s1 = MaxPool1D(pool_size=2)(conv2_s1)
    maxp2_s2 = MaxPool1D(pool_size=2)(conv2_s2)
    maxp2_s3 = MaxPool1D(pool_size=2)(conv2_s3)

    flat_p = Flatten()(maxp2_p)
    flat_s1 = Flatten()(maxp2_s1)
    flat_s2 = Flatten()(maxp2_s2)
    flat_s3 = Flatten()(maxp2_s3)
    flat_average = average([flat_s1, flat_s2, flat_s3])
    dot_1 = dot([flat_p, flat_average], -1)

    merged = concatenate([flat_p, flat_average, dot_1], axis=-1)

    dropped = Dropout(rate=0.2)(merged)
    dense1 = Dense(32)(dropped)

    dense = Dense(1)(dense1)
    output = Activation("sigmoid")(dense)

    if not compile:
        return input_p, input_s1, input_s2, input_s3, output

    model = Model([input_p, input_s1, input_s2, input_s3], output)
    model.compile(loss=reweight, optimizer='adam', metrics=['accuracy'])

    if print_summary:
        model.summary()

    return model


if __name__ == "__main__":

    batch_size = 128
    epochs = 75
    max_length = 25
    filters = 32
    num_seeds = 3
    embedding_size = 300
    callbacks = []
    dataset_feature_path = "../paraphrasing-data/cnn-3seeds-features.pickle"
    datasets = "../paraphrasing-data/crowdsourced"

    X1, X2, X3, X4, Y, T = get_or_create_features_file(dataset_feature_path, datasets, max_length)

    with tf.device('/gpu:1'):
        kfold = StratifiedKFold(Y, n_folds=5)
        ttn, tfp, tfn, ttp = 0, 0, 0, 0
        for train, test in kfold:
            model = create_model((max_length, embedding_size), filters=filters)
            model.fit([X1[train],X2[train],X3[train],X4[train]], Y[train], batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)
            scores = model.evaluate([X1[test],X2[test],X3[test],X4[test]], Y[test], verbose=1)
            y_pred_v = model.predict([X1[test],X2[test],X3[test],X4[test]])
            y_pred = (y_pred_v >= 0.5)
            tn, fp, fn, tp = print_statistics(Y[test], y_pred)
            ttn += tn
            tfp += fp
            tfn += fn
            ttp += tp
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

        print("tn:", ttn, "fp:", tfp, "fn:", tfn, "tp:", ttp)
        print("Accuracy:", (ttp + ttn) / (ttp + ttn + tfp + tfn))
