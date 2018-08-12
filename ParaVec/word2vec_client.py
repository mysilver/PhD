import requests
from functools import lru_cache


def n_similarity(sequence, sequence_2):
    resp = requests.post("http://localhost:4010/n_similarity", json={'seq_1': sequence, 'seq_2': sequence_2})
    return resp.json()["similarity"]


def wm_distance(sequence, sequence_2):
    resp = requests.post("http://localhost:4010/wm_distance", json={'seq_1': sequence, 'seq_2': sequence_2})
    return resp.json()["distance"]


def most_similar(words, ntop):
    resp = requests.post("http://localhost:4010/most_similar", json={'word': words, 'ntop': ntop})
    return resp.json()


@lru_cache(maxsize=15000)
def vector(word):
    resp = requests.post("http://localhost:4010/vector", json={'word': word})
    return resp.json()


def vocabulary():
    resp = requests.get("http://localhost:4010/vocabulary")
    return set(resp.json())


def most_similar_to_vector(vector, ntop=50):
    resp = requests.post("http://localhost:4010/most_similar_to_vector", json={'vector': vector, 'ntop': ntop})
    return resp.json()