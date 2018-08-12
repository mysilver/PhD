import os


def tab_splitor(line):
    return line.split('\t')


def read_paraphrased_tsv_files(directory: str, processor=None, reverse_columns: bool = True,
                               remove_html_tags: bool = True,
                               by_user: bool = False) -> dict:
    """
    Recursively reads the files in a directory and its sub-directories. 
    It is assumed that the first line starts with "# expression: " which indicates the expression which is paraphrased.
    :param by_user: 
    :param remove_html_tags: 
    :param reverse_columns: Reverse the order of columns in TSV files
    :param directory: The root directory where all files are located
    :param processor: 
    :return: dict
    """
    ret = {}
    for root, directories, filenames in os.walk(directory):
        for filename in filenames:
            f = os.path.join(root, filename)
            with open(f, 'rt') as file:
                print("reading " + f)
                file_dataset = []

                lines = file.readlines()
                expression = lines[0].replace("# expression:", "")
                if remove_html_tags:
                    expression = expression.replace("</b>", "").replace("<b class='fixed'>", "")
                if processor:
                    expression = processor(expression)
                user_dataset = []
                for i in range(1, len(lines)):

                    if len(lines[i].strip()) == 0:
                        if user_dataset:
                            if by_user:
                                file_dataset.append(user_dataset)
                            else:
                                file_dataset.extend(user_dataset)
                            user_dataset = []
                        continue

                    columns = tab_splitor(lines[i])
                    pc = []
                    for c in columns:
                        if processor:
                            pc.append(processor(c))
                        else:
                            pc.append(c)
                    if reverse_columns:
                        pc.reverse()
                    user_dataset.append(pc)

                if user_dataset:
                    if by_user:
                        file_dataset.append(user_dataset)
                    else:
                        file_dataset.extend(user_dataset)

                if expression not in ret:
                    ret[expression] = []
                ret[expression].extend(file_dataset)

    return ret


def read_corpus(file, splitor=tab_splitor, processor=None):
    """
    Reads and loads all items in a text file,
    :param file: a text file
    :param splitor: a function to determine how to split each line of the file 
    :param processor: a function to preprocess each line of the code
    :return: list of processed lines
    """
    ret = []
    with open(file, 'rt') as f:
        for line in f.readlines():
            temp = []
            for item in splitor(line):
                if processor:
                    temp.append(processor(item))
                else:
                    temp.append(item)
            ret.append(temp)
    return ret

# dataset = read_paraphrased_tsv_files("/media/may/Data/LinuxFiles/PycharmProjects/PhD/paraphrasing-data/crowdsourced", processor= str.strip)
# dataset = read_paraphrased_tsv_files("/media/may/Data/LinuxFiles/PycharmProjects/PhD/paraphrasing-data/crowdsourced",
#                            processor=str.strip, by_user=True)

# print(dataset)
