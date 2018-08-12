from utils.dataset import read_paraphrased_tsv_files
from utils.preprocess import remove_marks
from utils.service.ginger import correct

ginger_file = "../paraphrasing-data/ginger_correction.tsv"
ginger_error_map = {}


def load():
    with open(ginger_file, "rt") as f:
        for line in f.readlines():
            t = line.split("\t")
            ginger_error_map[remove_marks(t[0])] = (int(t[1]), bool(t[2]))


load()


def ginger_error_count(text):
    text = remove_marks(text)
    if text in ginger_error_map:
        return ginger_error_map[text]

    return 0, None


if __name__ == "__main__":

    datasets_path = "../paraphrasing-data/crowdsourced"
    datasets = read_paraphrased_tsv_files(datasets_path, by_user=False)

    correction_map = {}
    with open(ginger_file, "wt") as f:
        for index, expr in enumerate(datasets):
            paraphrases = datasets[expr]

            corrected, is_question = correct(expr, remove_case=True, sleep=True)
            if not corrected:
                corrected = {}
            else:
                print(expr.strip(), "==>", corrected)

            correction_map[expr] = corrected
            f.write(expr.strip() + '\t' + str(len(corrected)) + '\t' + str(is_question) + '\n')

            for para in enumerate(paraphrases):
                corrected, is_question = correct(para[1][0], sleep=True, remove_case=True)
                if not corrected:
                    corrected = {}
                else:
                    print(para[1][0].strip(), "==>", corrected)

                correction_map[para[1][0]] = corrected
                f.write(para[1][0].strip() + '\t' + str(len(corrected)) + '\t' + str(is_question) + '\n')
            print("Processed", 100 * index / len(datasets), "%")
