from functools import lru_cache

import requests


def similarity(sentence, sentence_2):
    try:
        resp = requests.post("http://localhost:4011/similarity", json={'seq_1': sentence, 'seq_2': sentence_2})
        return resp.json()["cosine_similarity"]
    except Exception as e:
        print("Error in similarity method:", sentence, sentence_2)
        raise e


@lru_cache(maxsize=500)
def vector(sentence):
    resp = requests.post("http://localhost:4011/vector", json={'sentence': sentence})
    return resp.json()
