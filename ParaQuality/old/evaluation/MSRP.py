import pandas as pd


def create_dataset(dataset_path, processor):
    dataset ={}

    with open(dataset_path, "rt") as f:
        for i,line in enumerate(f.readlines()):
            if i == 0:
                continue
            line = line.split('\t')
            label = line[0]
            p1 = line[3]
            p2 = line[4]
            if processor:
                p1 = processor(p1)
                p2 = processor(p2)

            if p1 not in dataset:
                dataset[p1] = []
            dataset[p1].append([p2, "invalid" if label == '0' else 'valid'])

    return dataset


def msrp_dataset(path, processor):
    return create_dataset(path, processor)


if __name__ == "__main__":
    dataset = msrp_dataset("/media/may/Data/ParaphrsingDatasets/MRPC/original/msr-para-train.tsv")

