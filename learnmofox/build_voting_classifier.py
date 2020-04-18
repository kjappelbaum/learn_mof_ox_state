# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from comet_ml import Experiment
from pathlib import Path
import time
import numpy as np
import click
from .tune_train import MLOxidationStates
import logging
import joblib
import os

RANDOM_SEED = 821996
STARTTIMESTRING = time.strftime("%Y%m%d-%H%M%S")
MIN_SAMPLES = 10


@click.command("cli")
@click.argument("xpath", type=click.Path(exists=True))
@click.argument("ypath", type=click.Path(exists=True))
@click.argument("xvalidpath", type=click.Path(exists=True))
@click.argument("yvalidpath", type=click.Path(exists=True))
@click.argument("xtestpath", type=click.Path(exists=True))
@click.argument("ytestpath", type=click.Path(exists=True))
@click.argument("modelpath", type=click.Path())
@click.argument("models", nargs=-1)
@click.argument("scaler", nargs=1)
def train_model(
    xpath,
    ypath,
    xvalidpath,
    yvalidpath,
    xtestpath,
    ytestpath,
    modelpath,
    models,
    scaler,
):
    if not os.path.exists(os.path.abspath(modelpath)):
        os.mkdir(os.path.abspath(modelpath))

    experiment = Experiment(project_name="mof-oxidation-states")
    experiment.log_asset(xpath)
    experiment.log_asset(ypath)
    experiment.log_asset(xvalidpath)
    experiment.log_asset(yvalidpath)
    experiment.log_asset(xtestpath)
    experiment.log_asset(ytestpath)

    train_stem = Path(xpath).stem
    ml_object = MLOxidationStates.from_x_y_paths(
        xpath=os.path.abspath(xpath),
        ypath=os.path.abspath(ypath),
        xvalidpath=os.path.abspath(xvalidpath),
        yvalidpath=os.path.abspath(yvalidpath),
        modelpath=os.path.abspath(modelpath),
        scaler=scaler,
        n=int(10),
        voting="soft",
        calibrate="istonic",
        experiment=experiment,
    )

    X_test = np.load(xtestpath)
    y_test = np.load(ytestpath)

    X_test = ml_object.scaler.transform(X_test)

    models_loaded = []

    for model in models:
        name = Path(model).stem
        model = joblib.load(model)
        models_loaded.append((name, model))
    votingclassifier = ml_object.calibrate_ensemble(
        models_loaded,
        ml_object.x_valid,
        ml_object.y_valid,
        ml_object.experiment,
        ml_object.voting,
        ml_object.calibrate,
    )

    votingclassifier_tuple = [("votingclassifier_" + train_stem, votingclassifier)]

    cores_test = ml_object.model_eval(
        votingclassifier_tuple, X_test, y_test, experiment, "test", modelpath
    )
    scores_train = ml_object.model_eval(
        votingclassifier_tuple, ml_object.x, ml_object.y, experiment, "train", modelpath
    )
    scores_valid = ml_object.model_eval(
        votingclassifier_tuple,
        ml_object.x_valid,
        ml_object.y_valid,
        experiment,
        "valid",
        modelpath,
    )


if __name__ == "__main__":
    train_model()  # pylint:disable=no-value-for-parameter
