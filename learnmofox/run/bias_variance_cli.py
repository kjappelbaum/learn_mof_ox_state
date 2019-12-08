# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import os
import sys

from glob import glob
import numpy as np
from pathlib import Path
from functools import partial
import click
import joblib
from mlxtend.evaluate import bias_variance_decomp
from dask.distributed import Client, LocalCluster
import pickle
from learnmofox.utils import make_if_not_exist
from learnmofox import utils

sys.modules['utils'] = utils
"""
CLI to get the bias-variance tradeoff for ensemble model and the base estimators.
"""


def bv_decomp_wrapper(model, xtrain, ytrain, xtest, ytest):
    name = Path(model[0]).stem
    modelobject = model[1]
    print(('working on model {}'.format(name)))

    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        modelobject,
        xtrain,
        ytrain,
        xtest,
        ytest,
        loss='0-1_loss',
        random_seed=821996,
        num_rounds=100,
    )

    result_dict = {
        'name': name,
        'avg_bias': avg_bias,
        'avg_expected_loss': avg_expected_loss,
        'avg_var': avg_var,
    }

    return result_dict


@click.command('cli')
@click.argument('modelpath', type=click.Path(exists=True))
@click.argument('xtrainpath', type=click.Path(exists=True))
@click.argument('ytrainpath', type=click.Path(exists=True))
@click.argument('xtestpath', type=click.Path(exists=True))
@click.argument('ytestpath', type=click.Path(exists=True))
@click.argument('outdir')
def main(modelpath, xtrainpath, ytrainpath, xtestpath, ytestpath, outdir):  # pylint:disable=too-many-arguments,too-many-locals
    """CLI"""
    scalerpath = os.path.join(modelpath, 'scaler_0.joblib')
    assert os.path.exists(scalerpath)
    scaler = joblib.load(scalerpath)

    make_if_not_exist(outdir)

    print('loading data')
    X_train = scaler.transform(np.load(xtrainpath))
    X_test = scaler.transform(np.load(xtestpath))

    y_train = np.load(ytrainpath).astype(np.int)
    y_test = np.load(ytestpath).astype(np.int)

    models = glob(os.path.join(modelpath, '*.joblib'))

    print('now starting dask and running the actual computation')
    global cluster
    global client
    cluster = LocalCluster(memory_limit='28GB', n_workers=4)
    client = Client(cluster)

    relevant_models = [[model, joblib.load(model)] for model in models if not 'scaler' in model]

    bvpartial = partial(bv_decomp_wrapper, xtrain=X_train, ytrain=y_train, xtest=X_test, ytest=y_test)
    futures = client.map(bvpartial, relevant_models)

    results = client.gather(futures)

    print('finished crunching, now dumping results')

    with open(os.path.join(outdir, 'bv_decomposition.pkl'), 'wb') as fh:
        pickle.dump(results, fh)


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
