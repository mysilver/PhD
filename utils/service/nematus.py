"""
I have changed Nematus server.py and nematus_style.py to fix the server bugs
cd '/media/may/New Volume/servers/nematus-master/prepuilt-models/wmt16_systems/en-de' 
 python /media/may/New\ Volume/servers/nematus-master/nematus/server.py -m model.npz --port 5001 -p 1 -v
./in    
"""
import json
import requests

from base.entities import ParaphraseData


class Translation:
    def __init__(self, tokens, score):
        self.score = score
        self.tokens = tokens


class Client(object):
    """
    A sample client for Nematus Server instances.

    Uses the Nematus API style, i.e., the server (`server.py`) must be started
    with `style=Nematus` to serve requests from this client.
    """

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.headers = {
            'content-type': 'application/json'
        }

    def _get_url(self, path='/'):
        return "http://{0}:{1}{2}".format(self.host, self.port, path)

    def translate(self, segment):
        """
        Returns the translation of a single segment.
        """
        payload = json.dumps({'segments': [segment],
                              "return_word_alignment": False,
                              "return_word_probabilities": True,
                              "suppress_unk": True,
                              "beam_width": 5,
                              "n_best": True})
        url = self._get_url('/translate')
        response = requests.post(url, headers=self.headers, data=payload)
        return [Translation(s['translation'], sum(s['word_probabilities'])) for s in response.json()['data']]

    def print_server_status(self):
        """
        Prints the server's status report.
        """
        url = self._get_url('/status')
        response = requests.get(url, headers=self.headers)
        print(json.dumps(response.json(), indent=4))


class NematusParaphraseGenerator:
    def __init__(self, src_translator_host='localhost', src_translator_port=5001, en_translator_host='localhost', en_translator_port=5002):
        self.en2deClient = Client(src_translator_host, src_translator_port)
        self.de2enClient = Client(en_translator_host, en_translator_port)

    def paraphrase(self, tokens):
        resp = self.en2deClient.translate(tokens)
        paraphrases = []
        para_set = set()
        input_text = " ".join(tokens)
        for p in resp:
            trans = self.de2enClient.translate(p.tokens)
            for t in trans:
                text = " ".join(t.tokens)
                if text not in para_set and input_text != text:
                    paraphrases.append(ParaphraseData(p.tokens, t.tokens, p.score * t.score))
                    para_set.add(text)
        return paraphrases


if __name__ == "__main__":
    paraphrases = NematusParaphraseGenerator().paraphrase("find some restaurants nearby".split())
    for p in paraphrases:
        print(p.score, p.target_tokens)
