from ParaQuality.weka import create_weka_arff_from
from utils.dataset import read_paraphrased_tsv_files
from utils.preprocess import remove_marks
from ParaQuality.features import OmniFeatureFunction, PeerFF


def independent_paraphrases(datasets_path):
    datasets = read_paraphrased_tsv_files(datasets_path, processor=lambda x: x)
    attrs = []
    labels = set()
    for expr in datasets:
        paraphrases = datasets[expr]
        for i, para in enumerate(paraphrases):
            text = para[0]
            label = para[1]
            labels.add(label)
            f = ff.extract(expr, text, i % 3)
            f.append(label)
            attrs.append(f)

    return attrs, labels


def normalize(datasets):
    for index, expr in enumerate(datasets):
        for para3 in datasets[expr]:
            for p in para3:
                p[1] = keep_one_label(p[1])

    return datasets


def keep_one_label(labels: str):

    if "valid" in labels and "invalid" not in labels:
        return "valid"

    if "translate" in labels:
        return "translate"
    if "answer" in labels:
        return "answer"
    if "cheating" in labels:
        return "cheating"
    if "answer" in labels:
        return "answer"
    if "divergence" in labels or "invalid" in labels:
        return "divergence"
    if "grammar" in labels:
        return "grammar"
    if "spelling" in labels:
        return "spelling"

    raise Exception("Label error:" + labels)


def dependent_paraphrases(datasets_path):
    datasets = read_paraphrased_tsv_files(datasets_path, processor=remove_marks, by_user=True)
    datasets = normalize(datasets)
    attrs = []
    labels = set()
    for index, expr in enumerate(datasets):
        paraphrases = datasets[expr]
        for i, para3 in enumerate(paraphrases):
            text_1 = para3[0][0]
            label_1 = para3[0][1]

            text_2 = para3[1][0]
            label_2 = para3[1][1]

            text_3 = para3[2][0]
            label_3 = para3[2][1]

            labels.add(label_1)
            labels.add(label_3)
            labels.add(label_3)

            f = ff.extract(expr, text_1, 1)
            f.extend(pff.extract(text_1, text_2, 0))
            f.extend(pff.extract(text_1, text_3, 0))
            f.append(label_1)
            attrs.append(f)

            f = ff.extract(expr, text_2, 2)
            f.extend(pff.extract(text_2, text_1, 0))
            f.extend(pff.extract(text_2, text_3, 0))
            f.append(label_2)
            attrs.append(f)

            f = ff.extract(expr, text_3, 3)
            f.extend(pff.extract(text_3, text_1, 0))
            f.extend(pff.extract(text_3, text_2, 0))
            f.append(label_3)
            attrs.append(f)
            # break
        # break
        print("Processed Expression", int(100 * (index + 1) / len(datasets)), "%")

    return attrs, labels


if __name__ == "__main__":
    # feature_log_file = "../paraphrasing-data/featuresFF.pickle"
    ff = OmniFeatureFunction()
    pff = PeerFF()

    dataset_feature_path = "../paraphrasing-data/para-feedback-dependent.arff"
    datasets_path = "../paraphrasing-data/merged_datasets"

    attributes, classes = dependent_paraphrases(datasets_path)
    create_weka_arff_from(attributes, classes, dataset_feature_path)
    # ff.save_logs()
