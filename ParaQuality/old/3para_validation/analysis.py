import numpy as np

from utils.dataset import read_paraphrased_tsv_files
from utils.preprocess import remove_marks

get_bin = lambda x, n: format(x, 'b').zfill(n)


def confusion_matrix_interpretation(array):
    array = np.array(array)
    array = np.reshape(array, (8, 8)).tolist()

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(8):
        for j in range(8):
            bits = get_bin(i, 3)
            predicted_bits = get_bin(j, 3)
            for k in range(3):
                if bits[k] == predicted_bits[k]:
                    if bits[k] == '1':
                        tp += array[i][j]
                    else:
                        tn += array[i][j]
                else:
                    if bits[k] == '1':
                        fn += array[i][j]
                    else:
                        fp += array[i][j]

    return tp, tn, fp, fn


if __name__ == "__main__":
    confusion_matrix_analysis = False
    dataset_statistics = True

    if confusion_matrix_analysis:
        confusion_matrix = "71   0   0   1   0   5   2  77\
        5   0   0   0   0   0   0   2\
        2   0   0   1   0   0   0  11\
        3   0   0   1   0   0   0  23\
        4   0   0   0   4   3   0  24\
        4   0   0   0   1   0   0  25\
        4   0   0   0   1   0   1  50\
        34   0   0   2   5  10  11 416"

        confusion_matrix = [int(a) for a in list(filter(None, confusion_matrix.split(" ")))]
        print(confusion_matrix_interpretation(confusion_matrix))

    if dataset_statistics:
        datasets = "../../paraphrasing-data/crowdsourced"
        dataset = read_paraphrased_tsv_files(datasets, processor=remove_marks, by_user=True)

        valid_1st = 0
        valid_2nd = 0
        valid_3rd = 0
        all3_valid = 0
        all3_invalid = 0
        counter = 0
        for i, expression in enumerate(dataset):
            for instance in dataset[expression]:
                counter += 1
                paraphrase_1 = instance[0][1] == 'valid'
                paraphrase_2 = instance[1][1] == 'valid'
                paraphrase_3 = instance[2][1] == 'valid'

                if paraphrase_1:
                    valid_1st += 1
                if paraphrase_2:
                    valid_2nd += 1
                if paraphrase_3:
                    valid_3rd += 1

                if paraphrase_1 and paraphrase_2 and paraphrase_3:
                    all3_valid += 1

                if not paraphrase_1 and not paraphrase_2 and not paraphrase_3:
                    all3_invalid += 1

        print("Total number of samples (each 3 paraphrases):", counter)
        print("1st valid:", valid_1st)
        print("2nd valid:", valid_2nd)
        print("3rd valid:", valid_3rd)

        print("All valid:", all3_valid)
        print("All invalid:", all3_invalid)
        print("The rest:", counter - all3_valid - all3_invalid)
