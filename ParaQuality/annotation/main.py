from flask import Flask, render_template, request, redirect, url_for

from utils.dataset import read_paraphrased_tsv_files

app = Flask(__name__)

datasets_path = "../../paraphrasing-data/merged_datasets"
datasets = read_paraphrased_tsv_files(datasets_path, by_user=False, processor=str.strip)
attrs = {}
attrs['index'] = 0
expressions = list(datasets.keys())
expressions.sort()


@app.route('/', methods=['GET'])
def index():
    expr = expressions[attrs['index']]
    return render_template("annotate.html", expressions=expressions, datasets=datasets, expression=expr)


@app.route('/statistics', methods=['GET'])
def statistics():
    counter = {"valid": 0,
               "divergence": 0,
               "spelling": 0,
               "grammar": 0,
               "cheating": 0,
               "misuse": 0,
               "translate": 0,
               "answer": 0}

    count = 0
    for expr in datasets:
        for sample in datasets[expr]:
            count += 1
            tags = sample[1].split(',')
            tags = list(filter(None, tags))
            tags.sort()
            lables = "-".join(tags)
            if len(tags) > 1:
                if lables in counter:
                    counter[lables] += 1
                else:
                    counter[lables] = 1

            for t in tags:
                if t:
                    counter[t] += 1

    keys = list(counter.keys())
    keys.sort(key=lambda x: len(x))

    counter['total'] = count
    return render_template("statistics.html", keys=keys, stats=counter)


@app.route('/', methods=['POST'])
def select_expression():
    attrs['index'] = expressions.index(request.form['expr'])
    return redirect(url_for('index'))


@app.route('/annotate', methods=['POST'])
def annotate():
    expr = expressions[attrs['index']]

    for p in datasets[expr]:
        exp = p[0]
        p.clear()
        p.append(exp)
        p.append("")

    for key in request.form:
        tag = key[:key.index('_')]
        para = key[key.index('_') + 1:]

        for p in datasets[expr]:
            if p[0] == para:
                p[1] = p[1] + tag + ","

    attrs['index'] = attrs['index'] + 1

    return redirect(url_for('index'))


@app.route('/save', methods=['POST'])
def save():
    new_dir = '../../paraphrasing-data/merged_datasets/'

    for i, exp in enumerate(expressions):
        with open(new_dir + "task-" + str(i) + ".tsv", 'wt') as f:
            f.write("# expression: " + exp + "\n")
            for index, sample in enumerate(datasets[exp]):
                if index % 3 == 0:
                    f.write("\n")
                f.write(sample[1] + "\t" + sample[0] + "\n")

    return redirect(url_for('index'))


app.run()
