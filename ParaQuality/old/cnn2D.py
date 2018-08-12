import os
import pickle
import queue
import random

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, Flatten, Dropout, Conv2D
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
    X, Y, T = [], [], []

    for i, expression in enumerate(dataset):
        # expression_vec = sentence_to_vec(expression, max_vec)
        seeds, samples = find_diverse_seeds(expression, dataset[expression])
        seed_vecs = []
        for s in seeds:
            features = sentence_to_vec(s, max_vec)
            seed_vecs.append(features)

        for instance in samples:
            paraphrase = instance[0]
            x0 = [sentence_to_vec(paraphrase, max_vec)]
            x0.extend(seed_vecs)
            X.append(x0)
            Y.append(0 if instance[1] == 'invalid' else 1)
            T.append(expression + "<==>" + paraphrase)
        print("Dataset Processed", int((i + 1) / len(dataset) * 100), "%")
    if save:
        pickle.dump((X, Y, T), open(save, 'wb'))

    return np.array(X), np.array(Y), np.array(T)


def get_or_create_features_file(dataset_feature_path, datasets, max_vec, shuffle=True):
    # Read the file in the following format:
    # expression    paraphrase  [valid/invalid]
    if os.path.isfile(dataset_feature_path):
        with open(dataset_feature_path, 'rb') as f:
            X, Y, T = pickle.load(f)

    else:
        dataset = read_paraphrased_tsv_files(datasets, processor=remove_marks)
        X, Y, T = to_vector(dataset, max_vec, save=dataset_feature_path)

    if shuffle:
        z = list(zip(X, Y, T))
        random.shuffle(z)
        X, Y, T = zip(*z)

    X = np.array(X)
    Y = np.array(Y)
    T = np.array(T)
    X = np.transpose(X, (0, 2, 1, 3))
    print("Input :", X.shape)
    print("Output :", Y.shape)
    return X, Y, T


def create_model(shape, filters, print_summary=True, compile=True):
    input_1 = Input(shape=shape)
    conv = Conv2D(filters=filters, strides=(2, 1), kernel_size=2)
    conv_out_1 = conv(input_1)

    # maxp_1 = MaxPool2D(pool_size=(2, 1))(conv_out_1)

    conv_2 = Conv2D(filters=int(filters / 2), kernel_size=2)
    conv2_out_1 = conv_2(conv_out_1)

    # maxp2_1 = MaxPool2D(pool_size=(2, 1))(conv2_out_1)
    flat = Flatten()(conv2_out_1)

    dropped = Dropout(rate=0.3)(flat)
    dense1 = Dense(32)(dropped)

    output = Dense(1, name='cnn_output', activation="sigmoid")(dense1)

    if not compile:
        return input_1, output

    model = Model(input_1, output)
    model.compile(loss=reweight, optimizer='adam', metrics=['accuracy'])

    if print_summary:
        model.summary()

    return model


if __name__ == "__main__":

    batch_size = 128
    epochs = 100
    max_length = 25
    filters = 32
    num_seeds = 3
    embedding_size = 300
    callbacks = []
    dataset_feature_path = "../paraphrasing-data/cnn-2D-features.pickle"
    datasets = "../paraphrasing-data/crowdsourced"

    X, Y, T = get_or_create_features_file(dataset_feature_path, datasets, max_length)

    with tf.device('/gpu:1'):
        kfold = StratifiedKFold(Y, n_folds=5)
        ttn, tfp, tfn, ttp = 0, 0, 0, 0
        for train, test in kfold:
            model = create_model((max_length,  num_seeds + 1, embedding_size), filters=filters)
            model.fit(X[train], Y[train], batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)
            scores = model.evaluate(X[test], Y[test], verbose=1)
            y_pred_v = model.predict(X[test])
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
