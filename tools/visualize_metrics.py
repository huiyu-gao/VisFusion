""" Partially copied from [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Jiaming Sun and Yiming Xie. """

import argparse
import json
import numpy as np
import os


def visualize(fname):
    key_names = ['AbsRel', 'AbsDiff', 'SqRel', 'RMSE', 'LogRMSE', 'r1', 'r2', 'r3', 'complete',
                 'acc(cm)', 'comp(cm)', 'chamfer(cm)', 'prec', 'recal', 'fscore']

    metrics = json.load(open(fname, 'r'))
    metrics = sorted([(scene, metric) for scene, metric in metrics.items()], key=lambda x: x[0])
    scenes = [m[0] for m in metrics]
    metrics = [m[1] for m in metrics]

    keys = metrics[0].keys()
    metrics1 = {m: [] for m in keys}
    for m in metrics:
        for k in keys:
            metrics1[k].append(m[k])

    for k in key_names:
        if k in metrics1:
            v = np.nanmean(np.array(metrics1[k]))
        else:
            v = np.nan
        print('%10s %0.3f' % (k, v))

    # save results
    with open(fname.split('.json')[0] + '.txt', 'w+') as f:
        text = 'Scene'
        for key_name in key_names:
            text += ', ' + key_name

        for i in range(len(scenes)):
            text += '\n' + scenes[i]
            for key_name in key_names:
                text += ', ' + str(metrics[i][key_name])
        f.write(text)


def main():
    parser = argparse.ArgumentParser(description="VisFusion ScanNet Testing")
    parser.add_argument("--model", required=True, metavar="FILE",
                        help="path to metrics file")
    args = parser.parse_args()

    rslt_file = os.path.join(args.model, 'metrics.json')
    visualize(rslt_file)


if __name__ == "__main__":
    main()
