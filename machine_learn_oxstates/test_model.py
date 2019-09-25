# -*- coding: utf-8 -*-
"""
This module is meant to be used to test the performance on a seperate holdout set
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import pandas as pd
import json
import click
from comet_ml import Experiment
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.calibration import calibration_curve
from mlxtend.evaluate import feature_importance_permutation
from functools import partial
from joblib import load
from mine_mof_oxstate.utils import read_pickle
from six.moves import range
from six.moves import zip
import concurrent.futures
from functools import partial


class NpEncoder(json.JSONEncoder):
    """https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def _permutation_score_base(model, X, y, cv, metric_func, shuffle=True, _=None):
    model_ = model
    avg_score = []
    if shuffle:
        X = np.random.shuffle(X)
        y = np.random.shuffle(y)
    for train, test in cv.split(X, y):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        model_.fit(X_train, y_train)  # <- this is currently not implemented ...
        avg_score.append(metric_func(y_test, model_.predict(X_test)))

    return np.mean(np.array(avg_score))


def permutation_test(model, X, y, rounds=50, metric_func=balanced_accuracy_score, max_workers=4):
    cv = StratifiedKFold(5)
    permuted_scores = []
    score = _permutation_score_base(model, X, y, cv, metric_func, shuffle=False)
    base_permutation_score = partial(_permutation_score_base, model=model, X=X, y=y, cv=cv, metric_func=metric_func)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for metric in executor.map(_permutation_score_base, range(rounds)):
            permuted_scores.append(metric)

    permuted_scores = np.array(permuted_scores)
    p_value = (np.sum(permuted_scores >= score) + 1.0) / (rounds + 1)
    return score, permuted_scores, p_value


def bootstrapped_metrics(  # pylint:disable=too-many-arguments
        model, X, y, scoring_funcs, n_rounds=200, seed=1234) -> list:
    """Get bootstrapped statistics for the metrics estimated with the callables in scoring_funcs

    Arguments:
        model {sklearn model} -- sklearn model that needs to be tests
        X {np.array} -- array with features
        y {np.array} -- array with labels
        scoring_funcs {list} -- list of tuples (name, callable), where the callable is a scoring function
            that takes y_hat and y and returns a float

    Keyword Arguments:
        n_rounds {int} -- number of bootstrap samples (default: {200})
        seed {int} -- random seed (default: {1234})


    Returns:
        list -- list of dictionaries of metrics
    """
    rng = np.random.RandomState(seed)

    sample_idx = np.arange(X.shape[0])

    metrics = []

    for _ in range(n_rounds):
        scores = {}
        bootstrap_idx = rng.choice(sample_idx, size=sample_idx.shape[0], replace=True)
        prediction = model.predict(X[bootstrap_idx])
        for metricname, metric in scoring_funcs:
            scores[metricname] = metric(y[bootstrap_idx], prediction)
        metrics.append(scores)

    return metrics


def return_scoring_funcs():
    """

    Returns:
        list -- list of tuples (name, scoring function)
    """
    f1_micro = partial(f1_score, average='micro')
    f1_macro = partial(f1_score, average='macro')
    precision = partial(precision_score, average='micro')
    recall = partial(recall_score, average='micro')
    metrics = [
        ('accuracy', accuracy_score),
        ('balanced_accuracy', balanced_accuracy_score),
        ('f1_micro', f1_micro),
        ('f1_macro', f1_macro),
        ('precision', precision),
        ('recall', recall),
    ]

    return metrics


def test_model(  # pylint:disable=too-many-arguments
        modelpath: str,
        scalerpath: str,
        Xpath: str,
        ypath: str,
        namepath: str,
        outpath: str,
        featurelabelpath: str = None,
):  # pylint:disable=too-many-locals
    """Takes a trained model and performes some tests on it and calculates statistics.

    Arguments:
        modelpath {str} -- path to sklearn model in .joblib file
        modelpath {str} -- path to the scaler object
        Xpath {str} -- path to features in npz file
        ypath {str} -- path to labels in npz file
        namepath {str} -- path to names in pickle 3 file
        outpath {str} -- path to which the evaluation metrics are written

    Keyword Arguments:
        featurelabelpath {str} -- path to a picklefile with a list of the feature names, if not None, feature importances are also estimates (default {None})
    """
    lower_quantile = 2.5 / 100
    upper_quantile = 97.5 / 100

    experiment = Experiment(api_key=os.getenv('COMET_API_KEY', None), project_name='mof-oxidation-states')
    experiment.add_tag('model evaluation')

    model = load(modelpath)
    scaler = load(scalerpath)
    X = np.load(Xpath)
    X = scaler.transform(X)
    y = np.load(ypath)
    experiment.log_dataset_hash(X)
    names = read_pickle(namepath)

    scores = bootstrapped_metrics(model, X, y, scoring_funcs=return_scoring_funcs())

    df_metrics = pd.DataFrame(scores)

    means = df_metrics.mean().values
    medians = df_metrics.median().values
    lower = df_metrics.quantile(lower_quantile).values
    upper = df_metrics.quantile(upper_quantile).values
    stds = df_metrics.std().values

    cv = StratifiedKFold(10)
    # balanced_accuracy, balanced_acc_permutation_scores, balanced_accuracy_pvalue = permutation_test(
    #     model, X, y
    # )
    #
    metrics_dict = {}
    #
    # metrics_dict["balanced_accuracy_cv"] = balanced_accuracy
    # metrics_dict[
    #     "balanced_accuracy_permutation_scores"
    # ] = balanced_acc_permutation_scores
    # metrics_dict["balanced_accuracy_p_value"] = balanced_accuracy_pvalue

    prediction = model.predict(X)
    misclassified = np.where(y != prediction)
    misclassified_w_prediction_true = [(names[i], prediction[i], y[i]) for i in list(misclassified[0])]

    metrics_dict['misclassified'] = misclassified_w_prediction_true
    experiment.log_metric('misclassified', misclassified)
    if featurelabelpath is not None:
        feature_labels = read_pickle(featurelabelpath)
        imp_vals, imp_all = feature_importance_permutation(
            predict_method=model.predict,
            X=X,
            y=y,
            metric='accuracy',
            num_rounds=20,  # to get some errorbars
            seed=1,
        )
        importance_error = np.std(imp_all, axis=-1)
        importance_metrics = [
            (name, value, error) for name, value, error in zip(feature_labels, imp_vals, importance_error)
        ]
        experiment.log_metric('feature_importances', importance_metrics)
        metrics_dict['feature_importances'] = importance_metrics

    for i, column in enumerate(df_metrics.columns.values):
        metrics_dict[column] = (means[i], medians[i], stds[i], lower[i], upper[i])
        print((column, means[i], '_'.join([column, 'mean'])))
        experiment.log_metric('_'.join([column, 'mean']), means[i])
        experiment.log_metric('_'.join([column, 'median']), medians[i])
        experiment.log_metric('_'.join([column, 'lower']), lower[i])
        experiment.log_metric('_'.join([column, 'std']), stds[i])

    # experiment.log_metrics("balanced_accuracy_cv", balanced_accuracy)
    # experiment.log_metrics("balanced_accuracy_p_value", balanced_accuracy_pvalue)
    # experiment.log_metrics("missclassified", misclassified_w_prediction_true)
    #
    # cc = calibration_curve(y, model.predict(X), n_bins=10)

    # metrics_dict["calibration_curve_true_probab"] = cc[0]
    # metrics_dict["calibration_curve_predicted_probab"] = cc[1]

    # now write a .json with metrics for DVC
    with open(os.path.join(outpath, 'test_metrics.json'), 'w') as fp:
        json.dump(metrics_dict, fp, cls=NpEncoder)


@click.command('cli')
@click.argument('modelpath')
@click.argument('scalerpath')
@click.argument('xpath')
@click.argument('ypath')
@click.argument('namepath')
@click.argument('outpath')
@click.argument('featurelabelpath')
def main(modelpath, scalerpath, xpath, ypath, namepath, outpath, featurelabelpath):  # pylint:disable=too-many-arguments
    test_model(modelpath, scalerpath, xpath, ypath, namepath, outpath, featurelabelpath)


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
