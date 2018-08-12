from functools import lru_cache
from queue import PriorityQueue

import numpy as np
from gensim.models import KeyedVectors

print("Loading Word2Vec : Numberbatch")
model = KeyedVectors.load_word2vec_format('/media/may/Data/servers/conceptnet/numberbatch-en.txt', binary=False)

model.init_sims(replace=True)
vocab = set(model.vocab)
print("Word2vec is loaded")
print("vocabulary size : ", len(vocab))


@lru_cache(maxsize=10000)
def vector(word):
    if word in vocab:
        return model[word]

    return None


def vocab_filter(words, v):
    return list(filter(lambda x: x in v, words))


def n_similarity(tokens_1, tokens_2):
    tokens_1 = vocab_filter(tokens_1, vocab)
    tokens_2 = vocab_filter(tokens_2, vocab)
    if not tokens_1 or not tokens_2:
        return 0

    return model.n_similarity(tokens_1, tokens_2)


def most_similar_to_vector(vector, ntop):
    return model.similar_by_vector(np.array(vector), topn=ntop, restrict_vocab=None)


def most_similar(words, ntop=50):
    words = vocab_filter(words, vocab)
    if not words:
        return []

    ret = []
    t = [i for i in model.most_similar(positive=words, topn=ntop * 2) if i[0] in vocab]
    queue = PriorityQueue()
    for s in t:
        queue.put((s[1], s[0]))

    while not queue.empty():
        a = queue.get()
        ret.append((a[1], a[0]))
    ret.reverse()
    return ret


def get_vocabulary():
    return vocab


def wm_distance(sequence_1, sequence_2):
    sequence_1 = vocab_filter(sequence_1, vocab)
    sequence_2 = vocab_filter(sequence_2, vocab)
    return model.wmdistance(sequence_1, sequence_2)
