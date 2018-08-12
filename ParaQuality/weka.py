import pickle

from ParaQuality.features import OmniFeatureFunction, PeerFF


def create_weka_arff(dataset_feature_path, save_path):
    """
    Creates input for Weka
    :param tweets_dictionary: 
    :param save_path: 
    :return: 
    """

    with open(dataset_feature_path, 'rb') as f:
        X, Y, T = pickle.load(f)

    with open(save_path, 'wt') as f:
        f.write("@relation ParaQuality\n")
        for i in range(len(X[0][0])):
            f.write("@attribute attr_" + str(i + 1) + " numeric\n")

        f.write("@attribute score {0,1}\n\n@data\n")
        for i in range(len(X)):
            f.write(",".join([str(i) for i in X[i][0]]) + "," + str(Y[i]) + "\n")


def create_weka_arff_from(attributes, classes, save_path):
    with open(save_path, 'wt') as f:
        f.write("@relation ParaFeedback\n")
        for i in OmniFeatureFunction(load_from_logs=False).extractors:
            attr_num = 1
            if hasattr(i, 'attr_num'):
                attr_num = i.attr_num
            for j in range(attr_num):
                f.write("@attribute " + i.__name__ + str(j) + " numeric\n")

        for s in range(2):
            for i in PeerFF().extractors:
                attr_num = 1
                if hasattr(i, 'attr_num'):
                    attr_num = i.attr_num
                for j in range(attr_num):
                    f.write("@attribute Peer_" + i.__name__ + str(j) + "_" + str(s) + " numeric\n")

        classes = sorted(set(classes))
        f.write("@attribute label {" + ",".join(classes) + "}\n\n@data\n")
        for i in range(len(attributes)):
            f.write(",".join([str(i) for i in attributes[i]]) + "\n")


if __name__ == "__main__":
    create_weka_arff("../paraphrasing-data/features.pickle", "../paraphrasing-data/para_quality.arff")
