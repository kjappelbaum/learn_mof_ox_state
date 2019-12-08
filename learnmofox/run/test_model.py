# -*- coding: utf-8 -*-
# pylint:disable=too-many-arguments
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

from joblib import load
from mine_mof_oxstate.utils import read_pickle
from learnmofox.metrics import bootstrapped_metrics, return_scoring_funcs
from mlxtend.evaluate import feature_importance_permutation
from sklearn.calibration import calibration_curve
from six.moves import zip


class NpEncoder(json.JSONEncoder):
    """https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741"""

    def default(self, obj):  # pylint:disable=arguments-differ, method-hidden
        if isinstance(obj, np.integer):  # pylint:disable=no-else-return
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        return super(NpEncoder, self).default(obj)


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

    print('*** Loading data ***')
    model = load(modelpath)
    scaler = load(scalerpath)
    X = np.load(Xpath)
    X = scaler.transform(X)
    y = np.load(ypath)
    experiment.log_dataset_hash(X)
    names = read_pickle(namepath)

    print('*** Getting bootstrapped metrics, using 200 folds which takes some time ***')
    scores = bootstrapped_metrics(model, X, y, scoring_funcs=return_scoring_funcs())

    df_metrics = pd.DataFrame(scores)

    means = df_metrics.mean().values
    medians = df_metrics.median().values
    lower = df_metrics.quantile(lower_quantile).values
    upper = df_metrics.quantile(upper_quantile).values
    stds = df_metrics.std().values

    # print(
    #    " *** Running permuation test running 200 folds with 10 fold CV which takes forever ***"
    # )
    # cv = StratifiedKFold(10)
    # balanced_accuracy, balanced_acc_permutation_scores, balanced_accuracy_pvalue = permutation_test(
    #    model, X, y
    # )

    metrics_dict = {}

    # metrics_dict["balanced_accuracy_cv"] = balanced_accuracy
    # metrics_dict[
    #    "balanced_accuracy_permutation_scores"
    # ] = balanced_acc_permutation_scores
    # metrics_dict["balanced_accuracy_p_value"] = balanced_accuracy_pvalue

    prediction = model.predict(X)

    print(' *** Getting misclassified cases ***')
    misclassified = np.where(y != prediction)
    misclassified_w_prediction_true = [(names[i], prediction[i], y[i]) for i in list(misclassified[0])]

    metrics_dict['misclassified'] = misclassified_w_prediction_true
    experiment.log_metric('misclassified', misclassified)
    if featurelabelpath is not None:
        feature_labels = read_pickle(featurelabelpath)

        print('*** Getting feature importance ***')
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
        experiment.log_metric('_'.join([column, 'upper']), upper[i])
        experiment.log_metric('_'.join([column, 'std']), stds[i])

    # experiment.log_metrics("balanced_accuracy_cv", balanced_accuracy)
    # experiment.log_metrics("balanced_accuracy_p_value", balanced_accuracy_pvalue)
    # experiment.log_metrics("missclassified", misclassified_w_prediction_true)

    print(' *** Getting the calibration curve ***')
    cc = calibration_curve(y, model.predict(X), n_bins=10)

    metrics_dict['calibration_curve_true_probab'] = cc[0]
    metrics_dict['calibration_curve_predicted_probab'] = cc[1]

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
