# -*- coding: utf-8 -*-
"""
This module can be used to get a learning curve point with bootstrapped errorbars.
"""
import os

import click
import numpy as np
import pandas as pd

from learnmofox.metrics import bootstrapped_metrics, return_scoring_funcs
from learnmofox.utils import make_if_not_exist, summarize_data, training_calibrate


def setup_learning_curve_point(  # pylint:disable=too-many-arguments, too-many-locals
    trainxpath, trainypath, validxpath, validypath, points, basepath
):
    # Set up directories
    learning_curve_point_dir = os.path.join(
        basepath, "_".join(["model", str(int(points))])
    )
    make_if_not_exist(learning_curve_point_dir)

    datadir = os.path.join(learning_curve_point_dir, "data")
    make_if_not_exist(datadir)

    modeldir = os.path.join(learning_curve_point_dir, "model")
    make_if_not_exist(modeldir)

    # Select diverse subset
    x = np.load(trainxpath)
    y = np.load(trainypath)

    xvalid = np.load(validxpath)
    yvalid = np.load(validypath)

    xsummarized, ysummarized = summarize_data(x, y, points)
    xsummarized_valid, ysummarized_valid = summarize_data(xvalid, yvalid, points)

    np.save(os.path.join(datadir, "features"), xsummarized)
    np.save(os.path.join(datadir, "labels"), ysummarized)

    np.save(os.path.join(datadir, "features_valid"), xsummarized_valid)
    np.save(os.path.join(datadir, "labels_valid"), ysummarized_valid)

    return datadir, modeldir


def test_model(  # pylint:disable=too-many-arguments
    modelpath: str,
    xtrainpath: str,
    ytrainpath: str,
    xvalidpath: str,
    yvalidpath: str,
    xtestpath: str,
    ytestpath: str,
    outpath: str,
    numpoints: int,
    bootstraps: int,  # pylint:disable=unused-argument
):  # pylint:disable=too-many-locals

    print("Subsampling")
    datadir, modeldir = setup_learning_curve_point(
        xtrainpath, ytrainpath, xvalidpath, yvalidpath, numpoints, outpath
    )

    print("Training on smaller training set")
    model, scaler = training_calibrate(
        modelpath,
        os.path.join(datadir, "features.npy"),
        os.path.join(datadir, "labels.npy"),
        os.path.join(datadir, "features_valid.npy"),
        os.path.join(datadir, "labels_valid.npy"),
        modeldir,
        scaler="standard",
        voting="soft",
        calibration="isotonic",
    )

    print("Getting bootstrap metric on the training set")
    X = scaler.transform(np.load(os.path.join(datadir, "features.npy")))
    y = np.load(os.path.join(datadir, "labels.npy"))

    scores = bootstrapped_metrics(model, X, y, scoring_funcs=return_scoring_funcs())
    df_metrics = pd.DataFrame(scores)
    df_metrics.to_csv(
        os.path.join(outpath, "bootstrapped_metrics_train.csv"), index=False
    )

    # On the holdout set
    print("Getting bootstrap metrics on the test set")
    X = scaler.transform(np.load(xtestpath))
    y = np.load(ytestpath)
    scores = bootstrapped_metrics(model, X, y, scoring_funcs=return_scoring_funcs())
    df_metrics = pd.DataFrame(scores)
    df_metrics.to_csv(
        os.path.join(outpath, "bootstrapped_metrics_test.csv"), index=False
    )


@click.command("cli")
@click.argument("modelpath")
@click.argument("xtrainpath")
@click.argument("ytrainpath")
@click.argument("xvalidpath")
@click.argument("yvalidpath")
@click.argument("xtestpath")
@click.argument("ytestpath")
@click.argument("outpath")
@click.argument("numpoints", type=int)
@click.argument("bootstraps", type=int, default=200)
def main(
    modelpath,
    xtrainpath,
    ytrainpath,
    xvalidpath,
    yvalidpath,
    xtestpath,
    ytestpath,
    outpath,
    numpoints,
    bootstraps,
):  # pylint:disable=too-many-arguments
    test_model(
        modelpath,
        xtrainpath,
        ytrainpath,
        xvalidpath,
        yvalidpath,
        xtestpath,
        ytestpath,
        outpath,
        numpoints,
        bootstraps,
    )


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
