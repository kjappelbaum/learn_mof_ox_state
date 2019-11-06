# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import sys
import os
import click
import pickle
import joblib
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import concurrent.futures
import numpy as np
from functools import partial
from sklearn.preprocessing import StandardScaler
from mlxtend.evaluate import feature_importance_permutation
import shap
from learnmofox import utils, summarize_data
from learnmofox.utils import read_pickle, make_if_not_exist

sys.modules['utils'] = utils

POINTS = 5000


def permuation_importance_wrapper(datatuple, model, rounds, metric=balanced_accuracy_score):
    X = datatuple[0]
    y = datatuple[1]
    name = datatuple[2]

    imp_vals, imp_all = feature_importance_permutation(
        predict_method=model.predict,
        X=X,
        y=y,
        metric=metric,
        num_rounds=rounds,
        seed=821996,
    )

    result_dict = {'set': name, 'imp_vals': imp_vals, 'imp_all': imp_all}
    return result_dict


@click.command('cli')
@click.argument('model', type=click.Path(exists=True))
@click.argument('xtrainpath', type=click.Path(exists=True))
@click.argument('ytrainpath', type=click.Path(exists=True))
@click.argument('xtestpath', type=click.Path(exists=True))
@click.argument('ytestpath', type=click.Path(exists=True))
@click.argument('featurenamespath', type=click.Path(exists=True))
@click.argument('outpath', type=click.Path())
@click.argument('rounds', type=int, default=50)
@click.argument('points', type=int, default=5000)
@click.option('--use_shap', is_flag=True)
def main(  # pylint:disable=too-many-arguments, too-many-locals
        model,
        xtrainpath,
        ytrainpath,
        xtestpath,
        ytestpath,
        featurenamespath,
        outpath,
        rounds,
        points,
        use_shap,
):
    # load model and data and also scale the data
    print('loading model and data')
    model = joblib.load(model)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.load(xtrainpath))
    y_train = np.load(ytrainpath)

    X_test = scaler.transform(np.load(xtestpath))
    y_test = np.load(ytestpath)

    X_train, y_train = summarize_data(X_train, y_train, points)
    X_test, y_test = summarize_data(X_test, y_test, points)

    # load the feature names
    feature_names = read_pickle(featurenamespath)

    permuation_importance_partial = partial(permuation_importance_wrapper, rounds=rounds, model=model)

    sets = [(X_train, y_train, 'train'), (X_test, y_test, 'test')]

    if not use_shap:
        print('starting permutation feature importance')
        # We do permutation feature importance for rounds rounds, using balanced accuracy as metric
        with concurrent.futures.ProcessPoolExector(max_workers=2) as executor:
            results = []
            for result in executor.map(permuation_importance_partial, sets):
                results.append(result)

        make_if_not_exist(outpath)
        with open(os.path.join(outpath, 'permutation_feature_importance.pkl'), 'wb') as fh:
            pickle.dump(fh, results)

    else:
        print('starting SHAP feature importance')
        make_if_not_exist(outpath)
        explainer = shap.KernelExplainer(
            model, X_train)  # note that we use the training set as the background dataset to integrate out features
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame()
        shap_values_df['feature'] = feature_names

        for i, shap_value in enumerate(shap_values):
            # Computing average impact of each feature in on model output (mean(abs(shap_values)) / per fold
            abs_mean_shap_values = np.mean(np.abs(shap_value), axis=0)
            expected_value = (explainer.expected_value[i] if explainer.expected_value[i] is not None else None)
            shap_values_df['shap_value_target_{}'.format(str(i))] = abs_mean_shap_values
            shap_values_df['expected_value_target_{}'.format(str(i))] = expected_value

        joblib.dump(explainer, os.path.join(outpath, 'shap_explainer'))
        shap_values_df.to_csv(os.path.join(outpath, 'shap_df.csv'), index=False)


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
