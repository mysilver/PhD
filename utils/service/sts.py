"""
/media/may/Data/servers/sts $ java -cp semantic-text-similarity-1.0-SNAPSHOT-jar-with-dependencies.jar semeval.PrebuiltModel

"""
from functools import lru_cache

import requests


class Client(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.headers = {
            'content-type': 'application/json'
        }

    def _get_url(self, path='/similarity'):
        return "http://{0}:{1}{2}".format(self.host, self.port, path)

    def similarity(self, sentence1, sentence2):
        url = self._get_url()
        response = requests.post(url, headers=self.headers, json={"sentence1": sentence1, "sentence2": sentence2})

        if not response.ok:
            print("STS error ", sentence1, sentence2)
            return 0

        content = response.content
        return float(content)


class StsSimilarity:
    def __init__(self, host='localhost', port=5005):
        self.client = Client(host, port)

    @lru_cache(maxsize=500)
    def similarity(self, sentence1, sentence2):
        return self.client.similarity(sentence1, sentence2)


# p = StsSimilarity().similarity("book", "books")
# print(p)
