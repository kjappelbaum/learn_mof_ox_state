# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import click
from learnmofox import utils
from learnmofox.metrics import permutation_test
import joblib
import numpy as np
import pickle
import sys
from sklearn.metrics import balanced_accuracy_score
from learnmofox.utils import summarize_data
sys.modules['utils'] = utils


@click.command('cli')
@click.argument('modelpath')
@click.argument('xpath')
@click.argument('ypath')
@click.argument('rounds', type=int, default=200)
@click.argument('points', type=int, default=100)
def main(modelpath, xpath, ypath, rounds, points):
    """CLI"""
    print('loading model and data')
    model = joblib.load(modelpath)
    X = np.load(xpath)
    y = np.load(ypath)
    X, y = summarize_data(X, y, points)

    print('starting actual permutation')
    score, permuted_scores, p_value = permutation_test(model, X, y, rounds=rounds, metric_func=balanced_accuracy_score)

    permutation_test_results = {
        'score': score,
        'permuted_scores': permuted_scores,
        'p_value': p_value,
    }

    with open('permutation_test_results.pkl', 'wb') as fh:
        pickle.dump(permutation_test_results, fh)


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
