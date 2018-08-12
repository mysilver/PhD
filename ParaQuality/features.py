import math
import os
import pickle
import jamspell
import editdistance
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from pyentrp import entropy as ent
from functools import lru_cache

from ParaQuality.correct_dataset import ginger_error_count
from utils.preprocess import tokenize, syllables, pos_tag, remove_marks
from utils.service import sentence2vec_client as sentence2vec
# from utils.service.ginger import correct
from utils.service.language_tool import SpellChecker
from utils.service.sts import StsSimilarity
from utils.service.word2vec_client import n_similarity, wm_distance
from utils.text import lcs
from langid.langid import LanguageIdentifier, model

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel('/media/may/Data/LinuxFiles/PycharmProjects/PhD/paraphrasing-data/en.spell.bin')
sts = StsSimilarity()
spellcheker = SpellChecker()


# ginger_map = pickle.load(
#     open("/media/may/Data/LinuxFiles/PycharmProjects/PhD/paraphrasing-data/ginger_correction_map.pickle", 'rb'))
#
# newmap = {}
# for k in ginger_map:
#     kn = remove_marks(k)
#     newmap[kn] = ginger_map[k]
# ginger_map = newmap

def do_not_preprocess():
    def wrapper(f):
        f.do_not_preprocess = True
        return f

    return wrapper


def attr_num(num):
    def wrapper(f):
        f.attr_num = num
        return f

    return wrapper


@attr_num(2)
def WordsFF(source, paraphrase, position):
    stokens = tokenize(source)
    ptokens = tokenize(paraphrase)

    word_num_diff = abs(len(stokens) - len(ptokens))
    letter_num_diff = len(source) - len(paraphrase)

    return [word_num_diff, letter_num_diff]


# def CommonWordsFF(source, paraphrase, position):
#     s = set(tokenize(source))
#     p = set(tokenize(paraphrase))
#     return 1 - len(p.difference(s)) / (len(p) + 1)


@attr_num(2)
def Entropy(source, paraphrase, position):
    s = set(tokenize(source))
    p = set(tokenize(paraphrase))
    word_diff_entropy = ent.shannon_entropy(" ".join(p.difference(s)))
    entropy = ent.shannon_entropy(paraphrase)
    return [entropy, word_diff_entropy]


@attr_num(3)
def PronounFF(source, paraphrase, position):
    s = set(tokenize(source))
    p = set(tokenize(paraphrase))

    i = {'i', 'me', 'my', 'mine', 'myself'}
    you = {'you', 'yours', 'yourself'}
    he = {'he', 'his', 'him', 'himself'}
    she = {'she', 'her', 'herself'}
    we = {'we', 'us', 'our', 'ours', 'ourselves'}
    they = {'they', 'their', 'them', 'themselves'}

    s_i = len(s.intersection(i)) > 1
    p_i = len(p.intersection(i)) > 1

    s_you = len(s.intersection(you)) > 1
    p_you = len(p.intersection(you)) > 1

    s_he = len(s.intersection(he)) > 1
    p_he = len(p.intersection(he)) > 1

    s_she = len(s.intersection(she)) > 1
    p_she = len(p.intersection(she)) > 1

    s_we = len(s.intersection(we)) > 1
    p_we = len(p.intersection(we)) > 1

    s_they = len(s.intersection(they)) > 1
    p_they = len(p.intersection(they)) > 1

    dangling_i = s_i == p_i
    dangling_you = s_you == p_you
    dangling_he = s_he == p_he
    dangling_she = s_she == p_she
    dangling_we = s_we == p_we
    dangling_they = s_they == p_they

    return [int(dangling_i), int(dangling_you), int(dangling_he or dangling_she or dangling_we or dangling_they)]


@attr_num(4)
def SemanticSimilarityFF(source, paraphrase, position):
    sts_sim = sts.similarity(source, paraphrase)
    sent2vec = sentence2vec.similarity(source, paraphrase)
    word2vec = n_similarity(tokenize(source), tokenize(paraphrase))
    wm = wm_distance(tokenize(source), tokenize(paraphrase))
    if math.isinf(wm):
        wm = 10

    return [sts_sim, sent2vec, word2vec, wm]


def DifferenceSimilarityFF(source, paraphrase, position):
    s = set(tokenize(source))
    p = set(tokenize(paraphrase))

    p = p.difference(s)
    s = s.difference(p)

    return n_similarity(list(p), list(s))


@attr_num(3)
@do_not_preprocess()
def SpellingFF(source, paraphrase, position):
    ss = spellcheker.check(source, ['TYPOS'], excludes_ids={'I_LOWERCASE'})
    ps = spellcheker.check(paraphrase, ['TYPOS'], excludes_ids={'I_LOWERCASE'})
    spell1 = len(ps) - len(ss)

    p = corrector.FixFragment(paraphrase)
    spell2 = editdistance.eval(p, paraphrase)

    s, _ = ginger_error_count(source)
    p, _ = ginger_error_count(paraphrase)
    spell3 = s / (p + 1)

    return [spell1, spell2, spell3]


def QuestionFF(source, paraphrase, position):
    _, q1 = ginger_error_count(source)
    _, q2 = ginger_error_count(paraphrase)

    return int(q1 == q2)


@do_not_preprocess()
def GrammarFF(source, paraphrase, position):
    ss = spellcheker.check(source, ['GRAMMAR'])
    ps = spellcheker.check(paraphrase, ['GRAMMAR'])
    gram = len(ps) / (len(ss) + 1)
    # gram2 = len(ginger_map.get(paraphrase, [])) / (len(ginger_map.get(source, [])) + 0.1)
    return gram  # [gram, gram2]


def CollocationFF(source, paraphrase, position):
    ss = spellcheker.check(source, ['COLLOCATIONS'])
    ps = spellcheker.check(paraphrase, ['COLLOCATIONS'])
    return len(ps) / (len(ss) + 1)


def ConfusedWordsFF(source, paraphrase, position):
    ss = spellcheker.check(source, ['CONFUSED_WORDS'])
    ps = spellcheker.check(paraphrase, ['CONFUSED_WORDS'])
    return len(ps) / (len(ss) + 1)


@do_not_preprocess()
def PunctuationFF(source, paraphrase, position):
    ss = spellcheker.check(source, ['TYPOGRAPHY', 'PUNCTUATION'])
    ps = spellcheker.check(paraphrase, ['TYPOGRAPHY', 'PUNCTUATION'])
    return len(ps) / (len(ss) + 1)


def SemanticErrorFF(source, paraphrase, position):
    ss = spellcheker.check(source, ['SEMANTICS'])
    ps = spellcheker.check(paraphrase, ['SEMANTICS'])
    return len(ps) / (len(ss) + 1)


def MiscErrorFF(source, paraphrase, position):
    ss = spellcheker.check(source, ['STYLE', 'MISC'])
    ps = spellcheker.check(paraphrase, ['STYLE', 'MISC'])
    return len(ps) / (len(ss) + 1)


def NonStandardPhraseFF(source, paraphrase, position):
    ss = spellcheker.check(source, ['NONSTANDARD_PHRASES'])
    ps = spellcheker.check(paraphrase, ['NONSTANDARD_PHRASES'])
    return len(ps) / (len(ss) + 1)


#
#
# def AutomatedReadabilityIndexFF(source, paraphrase, position):
#     def _measure(text):
#         spaces = text.count(' ') + 1
#         chars = len(text) - spaces
#         return 2.71 * chars / spaces + 0.5 * spaces / 1 - 21.43
#
#     return _measure(paraphrase) / (_measure(source) + 0.1)
#
#
# def GunningFogIndexFF(source, paraphrase, position):
#     def _measure(text):
#         words = tokenize(text)
#         complex_words = [w for w in words if len(syllables(w)) >= 3]
#         return 0.4 * (len(words) / 1 + 100 * len(complex_words) / (len(words) + 1))
#
#     return _measure(paraphrase) / (_measure(source) + 0.1)
#
#
# def ColemanLiauIndexFF(source, paraphrase, position):
#     def _measure(text):
#         S = 1
#         L = len(text)
#         return 0.0588 * L - 0.296 * S - 15.8
#
#     return _measure(paraphrase) / (_measure(source) + 0.1)
#
#
# def FleschKincaidReadabilityFF(source, paraphrase, position):
#     def _measure(text):
#         words = tokenize(text)
#         sylabs = len([w for w in words if len(syllables(w)) > 2])
#         words = len(words) + 1
#         return 206.835 - 1.015 * words / 1 - 84.4 * sylabs / words
#
#     return _measure(paraphrase) / (_measure(source) + 0.1)
#
#
# def SmogFF(source, paraphrase, position):
#     def _measure(text):
#         words = tokenize(text)
#         sylabs = len([w for w in words if len(syllables(w)) > 2])
#         return 1.0430 * math.sqrt(sylabs * 30 / 1) + 3.1291
#
#     return _measure(paraphrase) / (_measure(source) + 0.1)
#

@attr_num(3)
def EditDistanceFF(source, paraphrase, position):
    levenshtein = editdistance.eval(source, paraphrase)
    normalized_l = normalized_damerau_levenshtein_distance(source, paraphrase)
    normalized_d = normalized_damerau_levenshtein_distance(tokenize(source), tokenize(paraphrase))
    return [levenshtein, normalized_l, normalized_d]


@attr_num(3)
def EditPositionFF(source, paraphrase, position):
    lcs_v = lcs(source, paraphrase)
    num_edits = 0
    length_max_lcs = 0
    max_lcs_edit_center = -1
    if len(lcs_v) > 0:
        lcs_1 = lcs_v.pop()
        edits = paraphrase.split(lcs_1)
        edits = list(filter(None, edits))
        num_edits = len(edits)
        length_max_lcs = len(lcs_1) / len(source)
        # edit_center = (source.index(lcs_1) + len(lcs_1)) / (2 * len(paraphrase) + 1)
        max_lcs_edit_center = (source.index(lcs_1)) / (2 * len(source) + 1)

    return [num_edits, length_max_lcs, max_lcs_edit_center]


def TenseFF(source, paraphrase, position):
    """
    VB = 1
    VBG = 2
    VBN = 3
    VBZ = 4
    VBD = 5
    """

    def _pos_to_digit(pos):
        if pos == 'VB' or pos == 'VBP':
            return 1
        if pos == 'VBG':
            return 2
        if pos == 'VBN':
            return 3
        if pos == 'VBZ':
            return 4
        if pos == 'VBD':
            return 5

        return -1

    p = tokenize(paraphrase)
    tense = 0
    for t, tag in pos_tag(p):
        if 'VB' in tag:
            tense = max([tense, _pos_to_digit(tag)])
    return tense


def PositionFF(source, paraphrase, position):
    return position


@lru_cache(maxsize=500)
def LanguageFF(source, paraphrase, position):
    lang, p = identifier.classify(paraphrase)

    if lang == 'en':
        return p

    return 0


class OmniFeatureFunction:
    """
    This def is able to use all other feature functions 
    to generate features based on the source sentence and its paraphrase.
    """
    feature_dict = {}

    def __init__(self, logs_path=None, load_from_logs=True):

        self.extractors = [

            # Similarity Measures / LDA TOPIC Modeling
            SemanticSimilarityFF,
            # DifferenceSimilarityFF,
            # CommonWordsFF,

            # SubstringFF,  # 7
            EditDistanceFF,
            # LongestCommonSubstringFF,
            EditPositionFF,
            TenseFF,
            PronounFF,
            QuestionFF,
            # Informativeness
            # jargon
            WordsFF,
            # EntropyFF,
            # WordDifferenceEntropy,  # 17
            # Grammar
            SpellingFF,
            GrammarFF,
            CollocationFF,  # 20
            ConfusedWordsFF,  # 21
            PunctuationFF,  # 22
            SemanticErrorFF,  # 23
            MiscErrorFF,
            NonStandardPhraseFF,  # 25

            # Readability
            # AutomatedReadabilityIndexFF,
            # GunningFogIndexFF,
            # ColemanLiauIndexFF,
            # FleschKincaidReadabilityFF,
            # SmogFF,

            PositionFF,

            # Bias / Diversity
            # Missing Parameter
            # Jargon Distribution
            LanguageFF
        ]

        print("extractors:", self.extractors)
        self.logs_path = logs_path
        if load_from_logs and logs_path:
            if os.path.isfile(logs_path):
                with open(logs_path, 'rb') as f:
                    self.feature_dict = pickle.load(f)

    def extract(self, source, paraphrase, position):
        ret = []

        p_source = remove_marks(source)
        p_paraphrase = remove_marks(paraphrase)
        for extractor in self.extractors:

            if hasattr(extractor, 'do_not_preprocess'):
                value = extractor(source, paraphrase, position)
            else:
                value = extractor(p_source, p_paraphrase, position)
            # hash_id = hash(extractor.__name__ + source + paraphrase + str(position))

            # if hash_id in self.feature_dict:
            #     ret.extend(self.feature_dict[hash_id])
            #     continue

            if isinstance(value, list):
                ret.extend(value)
                # self.feature_dict[hash_id] = value
            else:
                ret.append(value)
                # self.feature_dict[hash_id] = [value]

        return ret

    def save_logs(self):
        with open(self.logs_path, 'wb') as f:
            pickle.dump(self.feature_dict, f)

    def extract_as_map(self, source, paraphrase, position):
        ret = {}
        for extractor in self.extractors:
            value = extractor.extract(source, paraphrase, position)
            ret[extractor.__name__] = value

        return ret


class PeerFF:
    """
    This def is able to use all other feature functions 
    to generate features based on the source sentence and its paraphrase.
    """

    def __init__(self):
        self.extractors = [

            # Similarity Measures / LDA TOPIC Modeling
            SemanticSimilarityFF,
            # SubstringFF,  # 7
            EditDistanceFF,
            # LongestCommonSubstringFF,
            EditPositionFF,
            TenseFF,
            LanguageFF,
            QuestionFF,

            # WordsFF,
            # PronounFF
        ]

        print("extractors:", self.extractors)

    def extract(self, source, paraphrase, position):
        ret = []
        for extractor in self.extractors:
            value = extractor(source, paraphrase, position)
            if isinstance(value, list):
                ret.extend(value)
            else:
                ret.append(value)

        return ret

    def extract_as_map(self, source, paraphrase, position):
        ret = {}
        for extractor in self.extractors:
            value = extractor(source, paraphrase, position)
            ret[extractor.__name__] = value

        return ret

# print(OmniFeatureFunction().extract("", "suggest a seattle hotel room for 2 tomorrow", 1))

# print(EditPositionFF("jump to the previous song", "my phone will jump to the previous song", 1))
# print(EditPositionFF("jump to previous song", "my phone will jump to the previous song that", 1))
