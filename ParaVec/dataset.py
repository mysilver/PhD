import json

import numpy as np
import math
import ParaVec.word2vec as word2vec
from nltk import ngrams
from nltk.tokenize import TweetTokenizer
from ParaVec.utils import *
import random

tknzr = TweetTokenizer()
vocab = word2vec.vocab


def tokenize(expression):
    for i in [3, 2]:
        tokens = tknzr.tokenize(expression)
        ngs = ngrams(tokens, i)
        for ng in ngs:
            phrs = "_".join(ng)
            if phrs in vocab and not math.isnan(word2vec.vector(phrs)[0]):
                expression = expression.replace(" ".join(ng), phrs)

    tokens = tknzr.tokenize(expression)
    return tokens


def to_embedding_sequence(line):
    ret = []
    for t in tokenize(line):
        if t in vocab:
            ret.append(word2vec.vector(t))
        else:
            ret.append(UNKNOWN_VECTOR)

    return ret


def create_word_embedding_dataset(dataset, output_path):
    print("Converting dataset '" + dataset + "'")
    with open(dataset) as f, open(output_path, 'wt') as output:
        for line in f:
            t = to_embedding_sequence(line)
            if not t:
                print("WARNING: ", line)
            seq = json.dumps(t)
            if "NaN" in seq:
                print("WARNING (NaN):", line)

            output.write(seq + "\n")

    print("Dataset is converted and stored in '" + output_path + "'")


def read_dataset(input_fileobject, output_fileobject, n=32):
    eof = False

    while True:
        ret = []
        ret2 = []
        # Read a line from the file: data
        while len(ret) < n:
            data = input_fileobject.readline()
            data2 = output_fileobject.readline()
            if not data and not data2:
                eof = True
                break
            elif not data or not data2:
                raise Exception("Datasets are not equal in size")

            l = json.loads(data)
            l2 = json.loads(data2)
            # add END token
            if l and l2:
                l.append(END_OF_SENTENCE_VECTOR)
                l2.append(END_OF_SENTENCE_VECTOR)

            ret.append(l)
            ret2.append(l2)

        # Yield the line of data
        if eof or not ret:
            break

        yield ret, ret2


def corpus_read(str_corora, normalize=True, shuffle=True, max_length=15):
    dataset = []
    for corpus in str_corora:
        with open(corpus) as corp:
            for line in corp:
                sentences = line.strip().split('\t')
                if len(sentences[0]) > 2 & len(sentences[0]) <= max_length and len(sentences[1]) > 2 & len(
                        sentences[1]) <= max_length:
                    if normalize:
                        dataset.append((normalizeString(sentences[0]), normalizeString(sentences[1])))
                    else:
                        dataset.append((sentences[0], sentences[1]))
    print("Number of samples:", len(dataset))
    if shuffle:
        random.shuffle(dataset)

    return dataset


def corpus_generator(str_corora, max_input_seq_length=15, input_vector_size=300,
                     max_target_seq_length=15, target_vector_size=300,
                     batch_size=32, shuffle=True):
    dataset = corpus_read(str_corora, shuffle=shuffle)
    batchs = math.ceil(len(dataset) / batch_size)
    while True:
        for batch in range(int(batchs)):
            start = batch * batch_size
            end = start + batch_size
            current = dataset[start:end]
            # process
            # The last batch might be less than batch_size
            real_batch_size = len(current)

            encoder_input_data = np.zeros((real_batch_size, max_input_seq_length, input_vector_size), dtype='float32')
            decoder_input_data = np.zeros((real_batch_size, max_target_seq_length, target_vector_size), dtype='float32')
            decoder_target_data = np.zeros((real_batch_size, max_target_seq_length, target_vector_size),
                                           dtype='float32')

            for i, sentences in enumerate(current):
                input_text = tokenize(sentences[0])
                target_text = tokenize(sentences[1])
                last_t = 0
                for t, word in enumerate(input_text):
                    if t < max_input_seq_length - 1:  # last token is 'dot'
                        if word in word2vec.vocab:
                            encoder_input_data[i, t] = word2vec.vector(word)
                        else:
                            encoder_input_data[i, t] = UNKNOWN_VECTOR
                        last_t = t
                encoder_input_data[i, last_t + 1] = END_OF_SENTENCE_VECTOR

                for t, word in enumerate(target_text):
                    # decoder_target_data is ahead of decoder_input_data by one timestep
                    if t < max_input_seq_length - 1:
                        if word in word2vec.vocab:
                            decoder_input_data[i, t] = word2vec.vector(word)
                        else:
                            decoder_input_data[i, t] = UNKNOWN_VECTOR
                        if t > 0:
                            # decoder_target_data will be ahead by one timestep
                            # and will not include the start character.
                            if word in word2vec.vocab:
                                decoder_target_data[i, t - 1] = word2vec.vector(word)
                            else:
                                decoder_target_data[i, t - 1] = UNKNOWN_VECTOR
                        last_t = t
                encoder_input_data[i, last_t + 1] = END_OF_SENTENCE_VECTOR
            yield [encoder_input_data, decoder_input_data], decoder_target_data


def para_vec_generator(input_dataset, output_dataset,
                       max_input_seq_length=15, input_vector_size=300,
                       max_target_seq_length=15, target_vector_size=300,
                       n=32, callback=None):
    with open(input_dataset) as input_f, open(output_dataset) as target_f:
        while True:
            try:
                d1, d2 = next(read_dataset(input_f, target_f, n))
                source_batch = list(d1)
                target_batch = list(d2)
                assert len(source_batch) == len(target_batch)
            except StopIteration:
                print()
                print("Finished processing the datasets in this epoch")
                input_f = open(input_dataset)
                target_f = open(output_dataset)
                continue

            # The last batch might be less than batch_size
            real_batch_size = len(source_batch)

            encoder_input_data = np.zeros((real_batch_size, max_input_seq_length, input_vector_size), dtype='float32')
            decoder_input_data = np.zeros((real_batch_size, max_target_seq_length, target_vector_size), dtype='float32')
            decoder_target_data = np.zeros((real_batch_size, max_target_seq_length, target_vector_size),
                                           dtype='float32')

            for i, (input_text, target_text) in enumerate(zip(source_batch, target_batch)):
                # if i % 1000 == 0:
                #     print(input_text, "<=>", target_text)

                for t, word in enumerate(input_text):
                    if t < max_input_seq_length:
                        encoder_input_data[i, t] = word
                for t, word in enumerate(target_text):
                    # decoder_target_data is ahead of decoder_input_data by one timestep
                    if t < max_input_seq_length:
                        decoder_input_data[i, t] = word
                        if t > 0:
                            # decoder_target_data will be ahead by one timestep
                            # and will not include the start character.
                            decoder_target_data[i, t - 1] = word
            if callback:
                callback()
            yield [encoder_input_data, decoder_input_data], decoder_target_data


def _read_dataset_example():
    with open("/media/may/Data/ParaphrsingDatasets/all/test.target.we") as f:
        for ret in read_dataset(f, f, 1):
            for expr in ret:
                l = json.loads(expr)
                print(l)


def _create_dataset_files():
    corpura = [
        "/media/may/Data/ParaphrsingDatasets/all/corpus/home.corpus",
        "/media/may/Data/ParaphrsingDatasets/all/corpus/mrpc.corpus",
        "/media/may/Data/ParaphrsingDatasets/all/corpus/MSR.corpus",
        "/media/may/Data/ParaphrsingDatasets/all/corpus/multinli.corpus",
        "/media/may/Data/ParaphrsingDatasets/all/corpus/P4P.corpus",
        "/media/may/Data/ParaphrsingDatasets/all/corpus/quora.corpus",
        "/media/may/Data/ParaphrsingDatasets/all/corpus/SLNI.corpus",
        "/media/may/Data/ParaphrsingDatasets/all/corpus/sts.corpus",
        "/media/may/Data/ParaphrsingDatasets/all/corpus/TE.corpus",
        # "/media/may/Data/ParaphrsingDatasets/all/corpus/askubuntu.corpus",
    ]

    dataset = corpus_read(corpura, normalize=True, shuffle=True)
    size = len(dataset)
    split = (0.99, 0.005, 0.05)  # training/dev/test splits
    train_start = 0
    dev_start = int(split[0] * size)
    test_start = int((split[0] + split[1]) * size)
    with open("training.source", 'wt') as s, open("training.target", 'wt') as t:
        for pairs in dataset[train_start:dev_start]:
            s.write(pairs[0] + '\n')
            t.write(pairs[1] + '\n')

    with open("dev.source", 'wt') as s, open("dev.target", 'wt') as t:
        for pairs in dataset[dev_start:test_start]:
            s.write(pairs[0] + '\n')
            t.write(pairs[1] + '\n')

    with open("test.source", 'wt') as s, open("test.target", 'wt') as t:
        for pairs in dataset[dev_start:size]:
            s.write(pairs[0] + '\n')
            t.write(pairs[1] + '\n')


if __name__ == '__main__':
    _create_dataset_files()
    # base = "/media/may/Data/ParaphrsingDatasets/all/"
    # create_word_embedding_dataset(base + "training.source", base + "training.source-UKN.we")
    # create_word_embedding_dataset(base + "training.target", base + "training.target-UKN.we")
