from collections import Counter

from utils.dataset import read_paraphrased_tsv_files
from utils.preprocess import remove_marks, tokenize, pos_tag
from nltk.corpus import stopwords
from nltk import ngrams

stopwords = set(stopwords.words('english'))

datasets = "../paraphrasing-data/crowdsourced"
dataset = read_paraphrased_tsv_files(datasets, processor=remove_marks)


def top_words():
    dictionary = Counter()
    for i, expression in enumerate(dataset):
        expr = set(tokenize(expression))
        for instance in dataset[expression]:
            paraphrase = instance[0]
            dictionary.update([t for t in tokenize(paraphrase) if t not in expr and t not in stopwords])

    for token in dictionary.most_common(50):
        print(token)


def top_ngrams(ngram):
    dictionary = Counter()
    for i, expression in enumerate(dataset):
        # expr = set(ngrams(tokenize(expression),ngram))
        for instance in dataset[expression]:
            paraphrase = instance[0]

            for t in ngrams(tokenize(paraphrase.lower()), ngram):
                phrase = " ".join(t)
                if phrase in expression.lower():
                    continue
                if type(t) is str:
                    tagged = pos_tag([t])
                else:
                    tagged = pos_tag(t)
                for t, tag in tagged:
                    if "VB" in tag:
                        dictionary.update([phrase])
                        break

    for token in dictionary.most_common(100):
        print(token)



top_words()