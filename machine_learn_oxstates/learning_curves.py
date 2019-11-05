# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import os
import pandas as pd
import numpy as np
import click

from comet_ml import Experiment
from sklearn.preprocessing import StandardScaler
from dask.distributed import Client, LocalCluster
from functools import partial
import itertools
from machine_learn_oxstates.utils import (
    setup_dummy,
    get_metrics_dict,
    make_if_not_exist,
    training_calibrate,
    summarize_data,
)

# Module to calculate learning curves.
# To run you need to:

# 1) Run the diverse set selection for a large set and get holdout set and validation set. This will stay fixed.
# 2) Then run the hyperparameter optimization on this large set.
# 3) After these steps, you can run the learning curve code. It will "summarize" the training set
#    from 1) using submodular selection and train the models on it and run the evaluation function
# 4) In addition to that, we will also "train" the uniform and stratified DummyClassifiers on the training set from 1)
#    and run the evaluation function on it to get the relevant baselin metrics.

# Quite a lot is written/read from disk which will increase the I/O overhead.


def model_evaluate(  # pylint:disable=too-many-arguments
        model,
        scaler,
        trainxpath,
        trainypath,
        holdoutxpath,
        holdoutypath,
        dummy=False):
    """Predicts using the model from training and holdout set and returns two dictionaries with the metrics"""
    if scaler is None:
        scaler = StandardScaler().fit(np.load(trainxpath))

    holdoutx = scaler.transform(np.load(holdoutxpath))
    holdouty = np.load(holdoutypath)

    trainx = scaler.transform(np.load(trainxpath))
    trainy = np.load(trainypath)

    trainy = trainy.astype(np.int)
    holdouty = holdouty.astype(np.int)

    # Get the predictions
    train_predict = model.predict(trainx)
    holdout_predict = model.predict(holdoutx)

    print((trainy.shape, holdouty.shape, train_predict.shape, holdout_predict.shape))
    # Get the metrics
    train_metrics = get_metrics_dict(trainy, train_predict, dummy)
    holdout_metrics = get_metrics_dict(holdouty, holdout_predict, dummy)

    # And return then
    return train_metrics, holdout_metrics


def setup_learning_curve_point(trainxpath, trainypath, points):
    # Set up directories
    learning_curve_point_dir = '_'.join(['model', str(int(points))])
    make_if_not_exist(learning_curve_point_dir)

    datadir = os.path.join(learning_curve_point_dir, 'data')
    make_if_not_exist(datadir)

    modeldir = os.path.join(learning_curve_point_dir, 'model')
    make_if_not_exist(modeldir)

    # Select diverse subset
    x = np.load(trainxpath)
    y = np.load(trainypath)

    xsummarized, ysummarized = summarize_data(x, y, points)

    np.save(os.path.join(datadir, 'features'), xsummarized)
    np.save(os.path.join(datadir, 'labels'), ysummarized)

    return datadir, modeldir


def orchestrate_training_point(  # pylint:disable=too-many-arguments
        modelpath,
        trainxpath,
        trainypath,
        validxpath,
        validypath,
        holdoutxpath,
        holdoutypath,
        points,
):
    # Setup directories and summarize data
    datadir, modeldir = setup_learning_curve_point(trainxpath, trainypath, points)

    # Now, we can train the model
    model, scaler = training_calibrate(
        modelpath,
        os.path.join(datadir, 'features.npy'),
        os.path.join(datadir, 'labels.npy'),
        validxpath,
        validypath,
        modeldir,
    )

    # Now, we evaluate
    train_metrics, holdout_metrics = model_evaluate(
        model,
        scaler,
        os.path.join(datadir, 'features.npy'),
        os.path.join(datadir, 'labels.npy'),
        holdoutxpath,
        holdoutypath,
    )
    train_metrics['set'] = 'train'
    train_metrics['train_points'] = train_metrics['points']
    holdout_metrics['train_points'] = train_metrics['points']
    holdout_metrics['set'] = 'holdout'

    # And we return the metrics
    return [train_metrics, holdout_metrics]


def learning_curve(  # pylint:disable=dangerous-default-value
        modelpath,
        trainxpath,
        trainypath,
        validxpath,
        validypath,
        holdoutxpath,
        holdoutypath,
        sizes=[
            100000,
            80000,
            60000,
            40000,
            20000,
            10000,
            5000,
            2000,
            1000,
            500,
        ],
):

    # Set up the dummy classifiers
    print('Getting dummy metrics')
    print(('using features at {} and labels at {}'.format(trainxpath, trainypath)))
    dummy_uniform, dummy_stratified = setup_dummy(trainxpath, trainypath)

    # Let's directly get the dummy metrics
    metrics = []
    train_metrics_uniform, hold_metrics_uniform = model_evaluate(
        dummy_uniform,
        None,
        trainxpath,
        trainypath,
        holdoutxpath,
        holdoutypath,
        dummy='uniform',
    )

    train_metrics_uniform['set'] = 'train'
    train_metrics_uniform['train_points'] = train_metrics_uniform['points']
    hold_metrics_uniform['train_points'] = train_metrics_uniform['points']
    hold_metrics_uniform['set'] = 'holdout'

    train_metrics_stratified, hold_metrics_stratified = model_evaluate(
        dummy_stratified,
        None,
        trainxpath,
        trainypath,
        holdoutxpath,
        holdoutypath,
        dummy='stratified',
    )

    train_metrics_stratified['set'] = 'train'
    train_metrics_stratified['train_points'] = train_metrics_stratified['points']
    hold_metrics_stratified['train_points'] = train_metrics_stratified['points']
    hold_metrics_stratified['set'] = 'holdout'

    # Load the model
    print('Now loading model and starting the loop')
    # model = joblib.load(modelpath)

    orchestrate = partial(
        orchestrate_training_point,
        modelpath,
        trainxpath,
        trainypath,
        validxpath,
        validypath,
        holdoutxpath,
        holdoutypath,
    )
    futures = client.map(orchestrate, sizes)

    results = client.gather(futures)
    results.append([
        train_metrics_stratified,
        train_metrics_uniform,
        hold_metrics_stratified,
        hold_metrics_uniform,
    ])
    results = list(itertools.chain.from_iterable(results))
    results.extend(metrics)

    df_results = pd.DataFrame(results)
    df_results.to_csv('learning_curve.csv', index=False)


# Below, we implement the CLI with Click


@click.command('cli')
@click.argument('modelpath', type=click.Path(exists=True))
@click.argument('trainxpath', type=click.Path(exists=True))
@click.argument('trainypath', type=click.Path(exists=True))
@click.argument('validxpath', type=click.Path(exists=True))
@click.argument('validypath', type=click.Path(exists=True))
@click.argument('holdoutxpath', type=click.Path(exists=True))
@click.argument('holdoutypath', type=click.Path(exists=True))
@click.argument('sizes', type=int, nargs=-1)
def main(  # pylint:disable=too-many-arguments
        modelpath,
        trainxpath,
        trainypath,
        validxpath,
        validypath,
        holdoutxpath,
        holdoutypath,
        sizes,
):
    print('*** Learning curve ***')
    print(('will calculate the learning curve for following training set sizes {}'.format(sizes)))

    # setup dask
    global cluster
    global client
    cluster = LocalCluster(n_workers=2, memory_limit='28GB')
    client = Client(cluster)

    print('starting learning curve function')
    learning_curve(
        modelpath,
        trainxpath,
        trainypath,
        validxpath,
        validypath,
        holdoutxpath,
        holdoutypath,
        sizes,
    )


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
