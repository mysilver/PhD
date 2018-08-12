import os
import pickle
import queue
import random

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, K, Activation, Flatten, MaxPooling1D
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

from ParaQuality.old.features import OmniFeatureFunction
from utils.dataset import read_paraphrased_tsv_files
from utils.preprocess import remove_marks


def reweight(y_true, y_pred, tp_weight=.3, tn_weight=.3, fp_weight=4, fn_weight=0.7):
    # Get predictions
    y_pred_classes = K.greater_equal(y_pred, 0.5)
    y_pred_classes_float = K.cast(y_pred_classes, K.floatx())

    # Get misclassified examples
    wrongly_classified = K.not_equal(y_true, y_pred_classes_float)
    wrongly_classified_float = K.cast(wrongly_classified, K.floatx())

    # Get correctly classified examples
    correctly_classified = K.equal(y_true, y_pred_classes_float)
    correctly_classified_float = K.cast(correctly_classified, K.floatx())

    # Get tp, fp, tn, fn
    tp = correctly_classified_float * y_true
    tn = correctly_classified_float * (1 - y_true)
    fp = wrongly_classified_float * y_true
    fn = wrongly_classified_float * (1 - y_true)

    # Get weights
    weight_tensor = tp_weight * tp + fp_weight * fp + tn_weight * tn + fn_weight * fn

    loss = K.binary_crossentropy(y_true, y_pred)
    weighted_loss = loss * weight_tensor
    return weighted_loss


def create_model(num_features=31, print_summary=False):
    input = Input(shape=(1, num_features))
    flat = MaxPooling1D(pool_size=1)(input)
    flat = Flatten()(flat)
    dense1 = Dense(8, activation='sigmoid', name="dense1")
    feature_importance = dense1(flat)
    dense_1 = Dense(32, name='dense')(feature_importance)

    dense = Dense(1, name='output')(dense_1)
    output = Activation("sigmoid")(dense)

    model = Model(input, output)

    model.compile(loss=reweight, optimizer='adam', metrics=['accuracy'])
    # model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    if print_summary:
        model.summary()

    return model


def find_diverse_seeds(expression, samples, seed_size=3):
    seeds = [expression]
    random.shuffle(samples)
    for s in samples:

        if len(seeds) == seed_size:
            break

        if s[1] == 'valid':
            seeds.append(s[0])

    return seeds, samples  # [s for s in samples if s[0] not in seeds]


def extract_features(dataset, save=None):
    X, Y, T = [], [], []
    ffs = OmniFeatureFunction()
    for i, expression in enumerate(dataset):
        seeds, instances = find_diverse_seeds(expression, dataset[expression], seed_size=1)
        index = 0
        for instance in instances:
            index += 1
            # try:
            paraphrase = instance[0]
            features = []
            for s in seeds:
                features.append(ffs.extract(s, paraphrase, index % 3))

            X.append(features)
            Y.append(0 if instance[1] == 'invalid' else 1)
            T.append(expression + "<==>" + paraphrase)
            # except Exception as e:
            #     print(expression, "-->", paraphrase)
            #     raise e
        print("Dataset Processed", int((i + 1) / len(dataset) * 100), "%")
    if save:
        pickle.dump((X, Y, T),
                    open(save, 'wb'))

    return np.array(X), np.array(Y), np.array(T)


def rank_features(target_tensor, input_tensor, xs, ys):
    scores = None
    for method in ['grad*input', 'saliency', 'intgrad', 'deeplift', 'elrp', 'occlusion']:
        attributions = de.explain(method, target_tensor * ys, input_tensor, xs)
        print(method, attributions)
        if scores is None:
            scores = attributions
        else:
            scores = scores + attributions
    return scores


def get_or_create_features_file(dataset_feature_path, datasets, shuffle=True):
    # Read the file in the following format:
    # expression    paraphrase  [valid/invalid]
    if os.path.isfile(dataset_feature_path):
        with open(dataset_feature_path, 'rb') as f:
            X, Y, T = pickle.load(f)

    else:
        dataset = read_paraphrased_tsv_files(
            datasets,
            processor=remove_marks)
        X, Y, T = extract_features(dataset, save=dataset_feature_path)

    if shuffle:
        z = list(zip(X, Y, T))
        random.shuffle(z)
        X, Y, T = zip(*z)

    X = np.array(X)
    X = np.nan_to_num(X)
    Y = np.array(Y)
    T = np.array(T)

    # if len(X.shape) == 3:
    # X = np.transpose(X, (0, 2, 1, 3))
    # X = np.reshape(X, (X.shape + (1,)))
    print("Input :", X.shape)
    print("Output :", Y.shape)
    return X, Y, T


def print_statistics(y_real, Y_pred):
    tn, fp, fn, tp = confusion_matrix(y_real, Y_pred).ravel()

    print("************************************")
    print("Class 0: ", tn + fp)
    print("Precision: ", tn / (tn + fn))
    print("Recall: ", tn / (tn + fp))
    print("------------------------------------")
    print("Class 1: ", tp + fn)
    print("Precision: ", tp / (tp + fp))
    print("Recall: ", tp / (tp + fn))
    print("------------------------------------")
    print("tn:", tn, "fp:", fp, "fn:", fn, "tp:", tp)
    return tn, fp, fn, tp


if __name__ == "__main__":

    batch_size = 128
    epochs = 500
    callbacks = []
    dataset_feature_path = "../paraphrasing-data/features.pickle"
    datasets = "../paraphrasing-data/crowdsourced"

    X, Y, T = get_or_create_features_file(dataset_feature_path, datasets)

    with tf.device('/gpu:1'):
        kfold = StratifiedKFold(Y, n_folds=5)
        ttn, tfp, tfn, ttp = 0, 0, 0, 0

        for train, test in kfold:
            model = create_model()
            model.fit(X[train], Y[train], batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)
            scores = model.evaluate(X[test], Y[test], verbose=0)
            y_pred_v = model.predict(X[test])
            y_pred = (y_pred_v >= 0.5)
            tn, fp, fn, tp = print_statistics(Y[test], y_pred)
            ttn += tn
            tfp += fp
            tfn += fn
            ttp += tp
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

            y_pred = y_pred.tolist()
            x_test = X[test]
            y_test = Y[test]
            t_test = T[test]

            q = queue.PriorityQueue()
            for i in range(len(y_pred_v)):
                pred_class = y_pred[i]
                real_class = y_test[i] == 1

                y = y_pred_v[i][0]
                y_r = y_test[i]

                if pred_class[0] != real_class:
                    q.put((- abs(y - y_r), y_r, y, t_test[i], ",".join([str(j) for j in x_test[i]])))

                if q.qsize() == 20:
                    break

            while not q.empty():
                score, real, stimated, text, features = q.get()
                # print(real, stimated, text)  # features

                # with DeepExplain(session=K.get_session()) as de:
                #     input_tensor = model.layers[0].input
                #     fModel = Model(inputs=input_tensor, outputs=model.layers[-2].output)
                #     fModel.summary()
                #     target_tensor = fModel(input_tensor)
                #     xs = X[0:1]
                #     ys = Y[0:1]
                #     rank_features(target_tensor, input_tensor, xs, ys)

        print("tn:", ttn, "fp:", tfp, "fn:", tfn, "tp:", ttp)
        print("Samples:", ttn + tfn + tfp + ttp)
        print("Accuracy:", (ttp + ttn) / (ttp + ttn + tfp + tfn))
