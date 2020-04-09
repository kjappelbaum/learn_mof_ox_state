# -*- coding: utf-8 -*-
# pylint:disable=too-many-arguments, too-many-locals, line-too-long, logging-fstring-interpolation

"""
This is to streamline the modeling process. One should be able to use only one module 
to reproduce the full work. 
Currently, this is overly complicated and which probably also makes it slow.
"""

from __future__ import absolute_import
from __future__ import print_function
from functools import partial
import time
import numpy as np
from collections import Counter
import concurrent.futures
import os
import json
import pickle
import logging
from typing import Tuple
from comet_ml import Experiment
from hyperopt import tpe, anneal, rand, mix, hp
from hpsklearn.estimator import hyperopt_estimator
from hpsklearn import components
from learnmofox.utils import VotingClassifier
from mlxtend.evaluate import BootstrapOutOfBag
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
import pandas as pd
from joblib import dump
import concurrent.futures
import click


def f1lossfn(y_true, y):
    f1lossfn=1-f1_score(y_true, y, average='macro')
    return f1lossfn


RANDOM_SEED = 821996
STARTTIMESTRING = time.strftime('%Y%m%d-%H%M%S')
MIN_SAMPLES = 10

classifiers = [
    (
        'sgd',
        partial(
            components.sgd,
            loss=hp.pchoice('loss', [(0.5, 'log'), (0.5, 'modified_huber')]),
        )
    ),
    ('svc', partial(
            components.svc_rbf, probability=True
        )),
    ('knn', components.knn),
   ('gradient_boosting', components.xgboost_classification),
    ('extra_trees', components.extra_trees),
    ('nb', components.gaussian_nb),
]

trainlogger = logging.getLogger('trainer')
trainlogger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(filename)s: %(message)s')
filehandler = logging.FileHandler(os.path.join(STARTTIMESTRING + '_train.log'))
filehandler.setFormatter(formatter)
trainlogger.addHandler(filehandler)


class MLOxidationStates:
    """Collects some functions used for training of the oxidation state classifier"""

    def __init__(
            self,
            X_train: np.array,
            y_train: np.array,
            X_valid : np.array, 
            y_valid: np.array, 
            n: int = 10,
            eval_method: str = 'kfold',
            scaler: str = 'standard',
            modelpath: str = 'models',
            max_evals: int = 250,
            voting: str = 'hard',
            calibrate: str = 'sigmoid',
            timeout: int = 600,
            max_workers: int = 16,
            experiment: Experiment = None
    ):  # pylint:disable=too-many-arguments

        self.x = X_train
        self.y = y_train
        self.x_valid = X_valid
        self.y_valid = y_valid 
        self.experiment = experiment
        
        # We make sure that everything is logged on comet 
        assert isinstance(experiment, Experiment)

        assert len(self.x) == len(self.y)
        assert len(self.x_valid) == len(self.y_valid)

        self.n = n
        self.eval_method = eval_method
        if scaler == 'robust':
            self.scalername = 'robust'
            self.scaler = RobustScaler()
        elif scaler == 'standard':
            self.scalername = 'standard'
            self.scaler = StandardScaler()
        elif scaler == 'minmax':
            self.scalername = 'minmax'
            self.scaler = MinMaxScaler()

        self.x = self.scaler.fit_transform(self.x)
        self.x_valid = self.scaler.transform(self.x_valid)

        classcounter = dict(Counter(self.y))
        trainlogger.info('the classdistribution is %s', classcounter)
        classes_to_keep = []
        for oxidationstate, count in classcounter.items():
            if count > MIN_SAMPLES:
                classes_to_keep.append(oxidationstate)
            else:
                trainlogger.warning(
                    'will drop class %s since it has not enough examples',
                    oxidationstate,
                )

        selected_idx = np.where(np.isin(self.y, classes_to_keep))[0]
        self.x = self.x[selected_idx]
        self.y = self.y[selected_idx]

        
        self.max_evals = max_evals
        self.voting = voting
        self.timeout = timeout
        self.timings = []
        self.modelpath = modelpath
        self.mix_ratios = {'rand': 0.15, 'tpe': 0.7, 'anneal': 0.15}
        self.max_workers = max_workers
        self.calibrate = calibrate
        self.classes = [1, 2, 3, 4, 5, 6, 7, 8]

        self.y = self.y.astype(np.int)
        self.y_valid = self.y.astype(np.int)

        trainlogger.info('intialized training class')
    
    @classmethod
    def from_x_y_paths(
            cls,
            xpath: str,
            ypath: str,
            xvalidpath: str,
            yvalidpath: str,
            modelpath: str,
            scaler: str,
            n: int,
            voting: str,
            calibrate: str,
            experiment: Experiment
    ):
        """Constructs a MLOxidationStates object from filepaths"""
        x = np.load(xpath, allow_pickle=True)
        y = np.load(ypath, allow_pickle=True)
        xvalid = np.load(xvalidpath, allow_pickle=True)
        yvalid = np.load(yvalidpath, allow_pickle=True)

        return cls(
            x,
            y,
            xvalid,
            yvalid,
            n=n,
            scaler=scaler,
            voting=voting,
            calibrate=calibrate,
            modelpath=modelpath,
            experiment=experiment
        )

    @staticmethod
    def tune_fit(  # pylint:disable=dangerous-default-value
            models: list,
            X: np.ndarray,
            y: np.ndarray,
            experiment: Experiment,
            max_evals: int = 400,
            timeout: int = 10 * 60,
            mix_ratios: dict = {
                'rand': 0.1,
                'tpe': 0.8,
                'anneal': 0.1
            },
            n: int = 10,
    ) -> list:
        """Tune model hyperparameters using hyperopt using a mixed strategy.
        Make sure when using this function that no data leakage happens.
        This data here should be seperate from training and test set.

        Arguments:
            models {list} -- list of models that should be optimized
            X {np.ndarray} -- features
            y {np.ndarray} -- labels
            max_evals {int} -- maximum number of evaluations of hyperparameter optimizations
            timeout {int} -- timeout in seconds after which the optimization stops
            mix_ratios {dict} -- dictionary which provides the ratios of the  different optimization algorithms
            n {int} -- number of folds
        Returns:
            list -- list of tuples (name, model) of optimized models
        """

        assert sum(list(mix_ratios.values())) == 1
        assert list(mix_ratios.keys()) == ['rand', 'tpe', 'anneal']

        trainlogger.debug('performing hyperparameter optimization')

        optimized_models = []

        mix_algo = partial(
            mix.suggest,
            p_suggest=[
                (mix_ratios['rand'], rand.suggest),
                (mix_ratios['tpe'], tpe.suggest),
                (mix_ratios['anneal'], anneal.suggest),
            ],
        )

        with experiment.train():
            partialml = partial(MLOxidationStates.train_one_model, X=X, y=y, max_evals=max_evals, mix_algo=mix_algo, timeout=timeout, n=n)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for name, m in executor.map(partialml, models):
                    trainlogger.info('trained {}'.format(name))
                    optimized_models.append((name, m))

        return optimized_models

    @staticmethod 
    def train_one_model(name_classifier, X: np.array, y: np.array, mix_algo, max_evals: int, timeout: int, n: int) -> Tuple: 
        name, classifier = name_classifier

        trainlogger.info("i'm using a timeout of {}".format(timeout))
        m = hyperopt_estimator(
            classifier=classifier('classifier'),
            algo=mix_algo,
            trial_timeout=timeout,
            preprocessing=[],
            max_evals=max_evals,
            loss_fn = f1lossfn, # f1 macro is probably more meaningfull than accuracy 
            # continuous_loss_fn = True, 
            seed=RANDOM_SEED,
        )
   
        trainlogger.info('training {}'.format(name))
        m.fit(X, y, cv_shuffle=True, n_folds=n) # hyperopt-sklearn takes care of the cross validations

        m.retrain_best_model_on_full_data(X, y)

        m = m.best_model()['learner']

        return (name, m)

    @staticmethod
    def calibrate_ensemble(
            models: list,
            X_valid: np.array, 
            y_valid: np.array,
            experiment: Experiment,
            voting: str = 'soft',
            calibrate: str = 'isotonic'
    ) -> Tuple[VotingClassifier, float]:
        """Collects base models into a voting classifier, trains it and then performs
        probability calibration

        Arguments:
            models {list} -- list of optimized base models
            X_valid {np.array} -- feature matrix
            y_valid {np.array} -- label vector

        Keyword Arguments:
            voting {str} -- voting mechanism (hard or soft) (default: {"soft"})
            n {int} -- number of CV folds for isotonic regression (default: {10})
            calibrate {str} -- probability calibration method (none, isotonic, sigmoid) (default: {isotonic})

        Returns:
            [CalibratedClassifierCV, float] -- [calibrated classifier and elapsed time]
        """
        trainlogger.debug('calibrating and building ensemble model')
        startime = time.process_time()

        models_sklearn = [(name, model) for name, model in models]

        # calibrate the base esimators
        with experiment.train():
            vc = VotingClassifier(models_sklearn, voting=voting)
            trainlogger.debug('now, calibrating the base base estimators')

            vc._calibrate_base_estimators(calibrate, X_valid, y_valid)  # pylint:disable=protected-access

        endtime = time.process_time()
        elapsed_time = endtime - startime

        return vc, elapsed_time

    @staticmethod
    def model_eval(
            models: list,
            x: np.array,
            y: np.array,
            experiment: Experiment,
            postfix: str = '_train',
            outdir_models: str = None,
    ):
        """Peforms a model evaluation on training and test set and dump the predictions with the actual values
        into an outout file

        Arguments:
            models {list} -- list of tuples with model name and model itself
            x {np.array} -- feature matrix 
            y {np.array} -- label vector 
            experiment {comet_ml.Experiment}
            postfix {str} -- string that will be attached to filename
            outdir_models {str} -- output directory for models
        """

        predictions = []

        trainlogger.debug('entered evaluation function')

        for name, model in models:
            postfix = '_'.join([postfix, name])
            outname_base_models = os.path.join(outdir_models, '_'.join([STARTTIMESTRING, postfix]))

            predict = model.predict(x)
            accuracy = accuracy_score(y, predict)

            f1_micro = f1_score(y, predict, average='micro')

            f1_macro = f1_score(y, predict, average='macro')

            balanced_accuracy= balanced_accuracy_score(y, predict)

            precision = precision_score(y, predict, average='micro')

            recall = recall_score(y, predict, average='micro')

            experiment.log_confusion_matrix(y, predict, title=postfix.strip('_'))

            prediction = {
                'model': name,
                'outname_base_models': outname_base_models,
                'accuracy' + postfix: accuracy,
                'f1_micro' + postfix: f1_micro,
                'f1_macro' + postfix: f1_macro,
                'balanced_accuracy' + postfix: balanced_accuracy,
                'precision' + postfix: precision,
                'recall' + postfix: recall,
                'points' + postfix: len(y),
                'n_features' + postfix: x.shape[0],
            }

            experiment.log_metrics(prediction)

            dump(model, outname_base_models + '.joblib')
            experiment.log_asset(outname_base_models + '.joblib')

            predictions.append(prediction)

        return predictions


@click.command('cli')
@click.argument('xpath', type=click.Path(exists=True))
@click.argument('ypath', type=click.Path(exists=True))
@click.argument('xvalidpath', type=click.Path(exists=True))
@click.argument('yvalidpath', type=click.Path(exists=True))
@click.argument('xtestpath', type=click.Path(exists=True))
@click.argument('ytestpath', type=click.Path(exists=True))
@click.argument('modelpath', type=click.Path())
@click.argument('scaler', default='standard')
@click.argument('voting', default='soft')
@click.argument('calibrate', default='isotonic')
@click.argument('n', default=10)
@click.argument('max_evals', default=250)
def train_model(
        xpath,
        ypath,
        xvalidpath, 
        yvalidpath, 
        xtestpath, 
        ytestpath,
        modelpath,
        scaler,
        voting,
        calibrate,
        n,
    max_evals
):
    if not os.path.exists(os.path.abspath(modelpath)):
        os.mkdir(os.path.abspath(modelpath))


    experiment = Experiment(project_name="mof-oxidation-states")
    experiment.log_parameter(name='scaler', value=scaler)
    experiment.log_parameter(name='n', value=n)
    experiment.log_parameter(name='voting', value=voting)
    experiment.log_parameter(name='calibrate', value=calibrate)
    experiment.log_parameter(name='max_evals', value=max_evals)
    experiment.log_asset(xpath)
    experiment.log_asset(ypath)
    experiment.log_asset(xvalidpath)
    experiment.log_asset(yvalidpath)
    experiment.log_asset(xtestpath)
    experiment.log_asset(ytestpath)

    ml_object = MLOxidationStates.from_x_y_paths(
        xpath=os.path.abspath(xpath),
        ypath=os.path.abspath(ypath),
        xvalidpath= os.path.abspath(xvalidpath),
        yvalidpath=os.path.abspath(yvalidpath),
        modelpath=os.path.abspath(modelpath),
        scaler=scaler,
        n=int(n),
        voting=voting,
        calibrate=calibrate,
        experiment=experiment
    )

    models = ml_object.tune_fit(
        classifiers,
        ml_object.x,
        ml_object.y,
        experiment=ml_object.experiment,
        max_evals=max_evals,
        timeout=1000,
        mix_ratios=ml_object.mix_ratios,
        n=ml_object.n,
    )
    
    X_test = np.load(xtestpath)
    y_test = np.load(ytestpath)

    X_test = ml_object.scaler.transform(X_test)

    dump(ml_object.scaler, os.path.join(modelpath, 'scaler.joblib'))
    experiment.log_asset(os.path.join(modelpath, 'scaler.joblib'))
    scores_test = ml_object.model_eval(models, X_test, y_test, experiment, 'test', modelpath)
    scores_train = ml_object.model_eval(models, ml_object.x, ml_object.y, experiment, 'train', modelpath)
    scores_valid = ml_object.model_eval(models, ml_object.x_valid,  ml_object.y_valid, experiment, 'valid', modelpath)


    votingclassifier = ml_object.calibrate_ensemble(
        ml_object.x_valid,
        ml_object.y_valid,
        ml_object.experiment,
        ml_object.voting,
        ml_object.calibrate,
    )

    votingclassifier_tuple = [('votingclassifier', votingclassifier)]

    cores_test = ml_object.model_eval(votingclassifier_tuple, X_test, y_test, experiment, 'test', modelpath)
    scores_train = ml_object.model_eval(votingclassifier_tuple, ml_object.x, ml_object.y, experiment, 'train', modelpath)
    scores_valid = ml_object.model_eval(votingclassifier_tuple, ml_object.x_valid, ml_object.y_valid, experiment, 'valid', modelpath)

    



if __name__ == '__main__':
    train_model()  # pylint:disable=no-value-for-parameter