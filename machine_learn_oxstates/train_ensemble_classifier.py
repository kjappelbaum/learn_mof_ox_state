# -*- coding: utf-8 -*-
# pylint:disable=too-many-arguments, too-many-locals, line-too-long, logging-fstring-interpolation
"""
Trains an ensemble classifier to predict the oxidation state
Produces a  outpath/train_metrics.json file for DVC

Note that it tries to fit the different folds in parallel using multiple processes, by default it
uses maximal 5 workers which is good e.g. require CV=5 or  CV=10 if you can run that many processes in parallel.
"""
from __future__ import absolute_import
from __future__ import print_function
from functools import partial
import time
from collections import Counter
import numpy as np
import os
import json
import pickle
import logging
from typing import Tuple
from comet_ml import Experiment
from hyperopt import tpe, anneal, rand, mix
from hpsklearn.estimator import hyperopt_estimator
from hpsklearn import components
from utils import VotingClassifier
from mlxtend.evaluate import BootstrapOutOfBag
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from ml_insights import SplineCalibratedClassifierCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import pandas as pd
from joblib import dump
import concurrent.futures
import click

RANDOM_SEED = 1234
STARTTIMESTRING = time.strftime('%Y%m%d-%H%M%S')
MIN_SAMPLES = 10

classifiers = [
    ('knn', components.knn),
    ('gradient_boosting', partial(components.gradient_boosting, loss='deviance')),
    ('extra_trees', components.extra_trees),
]

trainlogger = logging.getLogger('trainer')
trainlogger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(filename)s: %(message)s')
filehandler = logging.FileHandler(os.path.join('logs', STARTTIMESTRING + '_train.log'))
filehandler.setFormatter(formatter)
trainlogger.addHandler(filehandler)

VALID_SIZE = 0.3


class MLOxidationStates:
    """Collects some functions used for training of the oxidation state classifier"""

    def __init__(
            self,
            X: np.array,
            y: np.array,
            n: int = 10,
            max_size: int = None,
            eval_method: str = 'kfold',
            scaler: str = 'standard',
            metricspath: str = 'metrics',
            modelpath: str = 'models',
            max_evals: int = 50,
            voting: str = 'hard',
            calibrate: str = 'sigmoid',
            timeout: int = 600,
            oversampling: str = 'smote',
            max_workers: int = 5,
    ):  # pylint:disable=too-many-arguments

        self.x = X
        self.y = y
        assert len(self.x) == len(self.y)
        self.n = n
        self.eval_method = eval_method
        self.max_size = max_size
        if scaler == 'robust':
            self.scalername = 'robust'
            self.scaler = RobustScaler()
        elif scaler == 'standard':
            self.scalername = 'standard'
            self.scaler = StandardScaler()
        elif scaler == 'minmax':
            self.scalername = 'minmax'
            self.scaler = MinMaxScaler()

        self.bootstrap_results = []
        self.metrics = {}
        self.max_evals = max_evals
        self.voting = voting
        self.timeout = timeout
        self.timings = []
        self.metricspath = metricspath
        self.modelpath = modelpath
        self.mix_ratios = {'rand': 0.1, 'tpe': 0.8, 'anneal': 0.1}
        self.max_workers = max_workers
        self.calibrate = calibrate
        self.oversampling = oversampling

        trainlogger.info('intialized training class')

    @classmethod
    def from_x_y_paths(
            cls,
            xpath: str,
            ypath: str,
            modelpath: str,
            metricspath: str,
            scaler: str,
            n: int,
            voting: str,
            calibrate: str,
            max_size: int,
            oversampling: str,
    ):
        """Constructs a MLOxidationStates object from filepaths"""
        x = np.load(xpath, allow_pickle=True)
        y = np.load(ypath, allow_pickle=True)

        return cls(
            x,
            y,
            n=n,
            max_size=max_size,
            scaler=scaler,
            voting=voting,
            calibrate=calibrate,
            modelpath=modelpath,
            metricspath=metricspath,
            oversampling=oversampling,
        )

    @staticmethod
    def train_ensemble(
            models: list,
            X: np.array,
            y: np.array,
            voting: str = 'soft',
            calibrate: str = 'spline',
            valid_size: float = VALID_SIZE,
    ) -> Tuple[CalibratedClassifierCV, float]:
        """Collects base models into a voting classifier, trains it and then performs
        probability calibration

        Arguments:
            models {list} -- list of optimized base models
            X {np.array} -- feature matrix
            y {np.array} -- label vector

        Keyword Arguments:
            voting {str} -- voting mechanism (hard or soft) (default: {"soft"})
            n {int} -- number of CV folds for isotonic regression (default: {10})
            calibrate {str} -- probability calibration method (none, isotonic, sigmoid) (default: {soft})
            valid_size {float} -- fraction of the last part of the training set used for validation

        Returns:
            [CalibratedClassifierCV, float] -- [description]
        """
        trainlogger.debug('training ensemble model')
        models_sklearn = [(name, model) for name, model in models]
        # hyperopt uses by  default the last .2 percent as a validation set, we use the same convention here to do the
        # probability calibration
        # https://github.com/hyperopt/hyperopt-sklearn/blob/52a5522fae473bce0ea1de5f36bb84ed37990d02/hpsklearn/estimator.py#L268

        n_train = int(len(y) * (1 - valid_size))

        X_valid = X[n_train:]
        y_valid = y[n_train:]

        # calibrate the base esimators
        startime = time.process_time()
        models_calibrated = []
        for name, model_sklearn in models_sklearn:
            trainlogger.debug('calibrating  %s', name)
            model = model_sklearn
            trainlogger.debug(
                'the accuracay on the validation set before calibration is %s',
                accuracy_score(y_valid, model.predict(X_valid)),
            )
            calibrated = MLOxidationStates.calibrate_model(model, calibrate, X_valid, y_valid)
            models_calibrated.append((
                name,
                MLOxidationStates.calibrate_model(model, calibrate, X_valid, y_valid),
            ))
            trainlogger.debug(
                'the accuracay on the validation set after calibration is %s',
                accuracy_score(y_valid, calibrated.predict(X_valid)),
            )

        # due to the way this is implemented in sklearn, we cannot use the voting on prefit models

        vc = VotingClassifier(models_calibrated, voting=voting)

        endtime = time.process_time()
        elapsed_time = startime - endtime

        return vc, elapsed_time

    @staticmethod
    def calibrate_model(model, method: str, X_valid: np.array, y_valid: np.array):
        if method == 'isotonic':
            trainlogger.debug('calibrating using isotonic regression')
            calibrated = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
            calibrated.fit(X_valid, y_valid)
        elif method == 'sigmoid':
            trainlogger.debug('calibrating using sigmoid regression')
            calibrated = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
            calibrated.fit(X_valid, y_valid)
        elif method == 'none':
            trainlogger.debug('not calibrating')
            calibrated = model
        elif method == 'spline':
            trainlogger.debug('calibrating using spline calibration')
            calibrated = SplineCalibratedClassifierCV(model, cv='prefit')
            calibrated.fit(X_valid, y_valid)
        else:
            trainlogger.info(
                'could not understand choice for probability calibration method, will use sigmoid regression')
            calibrated = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
            calibrated.fit(X_valid, y_valid)

        return calibrated

    @staticmethod
    def tune_fit(  # pylint:disable=dangerous-default-value
            models: list,
            X: np.ndarray,
            y: np.ndarray,
            max_evals: int = 50,
            timeout: int = 10 * 60,
            mix_ratios: dict = {
                'rand': 0.1,
                'tpe': 0.8,
                'anneal': 0.1,
            },
            valid_size: float = VALID_SIZE,
    ) -> list:
        """Tune model hyperparameters using hyperopt using a mixed strategy.
        Make sure when using this function that no data leakage happens.
        This data here should be seperate from training and test set.

        Arguments:
            models {list} -- list of models that should be optimized
            X_valid {np.ndarray} -- features
            y_valid {np.ndarray} -- labels
            max_evals {int} -- maximum number of evaluations of hyperparameter optimizations
            timeout {int} -- timeout in seconds after which the optimization stops
            mix_ratios {dict} -- dictionary which provides the ratios of the  different optimization algorithms
            valid_size {float} -- fraction of the last part of the training set used for validation

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

        for name, classifier in models:
            m = hyperopt_estimator(
                classifier=classifier('classifier'),
                algo=mix_algo,
                trial_timeout=timeout,
                max_evals=max_evals,
                seed=RANDOM_SEED,
            )

            m.fit(X, y, valid_size=valid_size,
                  cv_shuffle=False)  # avoid shuffleing to have the same validation set for the ensemble stage

            # chose the model with best hyperparameters and train it
            m = m.best_model()['learner']

            n_train = int(len(y) * (1 - valid_size))
            X_train = X[:n_train]
            y_train = y[:n_train]

            m.fit(X_train, y_train)

            optimized_models.append((name, m))

        return optimized_models

    @staticmethod
    def get_bootstrap(X: np.ndarray, y: np.ndarray, n: int = 200):
        """Returns train, test, validation splits

        Arguments:
            X {np.ndarray} -- Feature matrix
            y {np.ndarray} -- Label vector
            n {int} -- number of bootstrap resamplings
        """

        assert len(X) == len(y)

        bs = BootstrapOutOfBag(n_splits=n, random_seed=RANDOM_SEED)

        oob = bs.split(np.arange(len(y)))

        return oob

    @staticmethod
    def get_train_test_split(X: np.ndarray, y: np.ndarray, n: int = 10):
        """Returns train, test, validation splits

        Arguments:
            X {np.ndarray} -- Feature matrix
            y {np.ndarray} -- Label vector
            n {int} -- number of split resamplings
        """
        bs = StratifiedKFold(n_splits=n, random_state=RANDOM_SEED)

        oob = bs.split(X, y)

        return oob

    @staticmethod
    def model_eval(
            models: list,
            xtrain: np.array,
            ytrain: np.array,
            xtest: np.array,
            ytest: np.array,
            postfix: str = 0,
            outdir_metrics: str = None,
            outdir_models: str = None,
    ):
        """Peforms a model evaluation on training and test set and dump the predictions with the actual values
        into an outout file

        Arguments:
            models {list} -- list of tuples with model name and model itself
            xtrain {np.array} -- feature matrix training set
            ytrain {np.array} -- label vector training set
            xtest {np.array} -- feature matrix test set
            ytest {np.array} -- label vector test set
            postfix {str} -- string that will be attached to filename
            outdir_metrics {str} -- output directory for metrics
            outdir_models {str} -- output directory for models
        """

        predictions = []

        trainlogger.debug('entered evaluation function')

        for name, model in models:
            outdir_metrics_verbose = os.path.join(os.path.join(outdir_metrics, 'verbose'))
            if not os.path.exists(outdir_metrics_verbose):
                os.mkdir(outdir_metrics_verbose)

            outname_base_metrics = os.path.join(outdir_metrics_verbose, '_'.join([STARTTIMESTRING, name, postfix]))
            outname_base_models = os.path.join(outdir_models, '_'.join([STARTTIMESTRING, name, postfix]))

            train_true = ytrain
            test_true = ytest

            train_predict = model.predict(xtrain)
            test_predict = model.predict(xtest)
            accuracy_train = accuracy_score(train_true, train_predict)
            accuracy_test = accuracy_score(test_true, test_predict)

            f1_micro_train = f1_score(train_true, train_predict, average='micro')
            f1_micro_test = f1_score(test_true, test_predict, average='micro')

            f1_macro_train = f1_score(train_true, train_predict, average='macro')
            f1_macro_test = f1_score(test_true, test_predict, average='macro')

            balanced_accuracy_train = balanced_accuracy_score(train_true, train_predict)
            balanced_accuracy_test = balanced_accuracy_score(test_true, test_predict)
            precision_train = precision_score(train_true, train_predict, average='micro')
            precision_test = precision_score(train_true, train_predict, average='micro')
            recall_train = recall_score(train_true, train_predict, average='micro')
            recall_test = recall_score(test_true, test_predict, average='micro')

            trainlogger.info(
                f'model {name}: accuracy test: {accuracy_test}, accuracy train: {accuracy_train} | f1 micro test {f1_micro_test}, f1 micro train {f1_micro_train}'
            )

            prediction = {
                'model': name,
                'postfix': postfix,
                'outname_base_models': outname_base_models,
                'outname_base_metrics': outname_base_metrics,
                'accuracy_train': accuracy_train,
                'accuracy_test': accuracy_test,
                'f1_micro_train': f1_micro_train,
                'f1_micro_test': f1_micro_test,
                'f1_macro_train': f1_macro_train,
                'f1_macro_test': f1_macro_test,
                'balanced_accuracy_train': balanced_accuracy_train,
                'balanced_accuracy_test': balanced_accuracy_test,
                'precision_train': precision_train,
                'precision_test': precision_test,
                'recall_train': recall_train,
                'recall_test': recall_test,
                'training_points': len(ytrain),
                'test_points': len(ytest),
            }

            arrays = {
                'train_true': train_true,
                'train_predict': train_predict,
                'test_predict': test_predict,
                'test_true': test_true,
            }

            arrays.update(prediction)

            predictions.append(arrays)

            with open(outname_base_metrics + '.pkl', 'wb') as fh:
                pickle.dump(arrays, fh)

            dump(model, outname_base_models + '.joblib')

        return predictions

    def train_eval_single(self, count_indx: tuple):
        """Peforms a optimize, train, evaluation loop on one fold

        Arguments:
            count_indx {tuple} -- (fold index, indices for training and test set)

        Returns:
            list -- list of dictionaries of model performance metrics
        """

        counter, tt_indices = count_indx

        trainlogger.debug('entered the function that trains one fold')
        all_predictions = []
        counter = str(counter)
        train, test = tt_indices

        scaler = self.scaler
        xtrain = self.x[train]
        ytrain = self.y[train]
        ytest = self.y[test]

        ytrain = ytrain.reshape(-1, 1)
        ytest = ytest.reshape(-1, 1)

        if self.oversampling == 'smote':
            trainlogger.debug('using smote oversampling')
            xtrain, ytrain = SMOTE(random_state=RANDOM_SEED).fit_resample(xtrain, ytrain)
            ytrain = ytrain.reshape(-1, 1)
        elif self.oversampling == 'borderlinesmote':
            trainlogger.debug('using BorderlineSMOTE oversamplign')
            xtrain, ytrain = BorderlineSMOTE(random_state=RANDOM_SEED).fit_resample(xtrain, ytrain)
            ytrain = ytrain.reshape(-1, 1)
        elif self.oversampling == 'adaysn':
            trainlogger.debug('using Adayn oversamplign')
            xtrain, ytrain = ADASYN(random_state=RANDOM_SEED).fit_resample(xtrain, ytrain)
            ytrain = ytrain.reshape(-1, 1)

        xtrain = scaler.fit_transform(xtrain)

        trainlogger.debug('the training set has shape %s', xtrain.shape)

        # save the latest scaler so we can use it later with latest model for
        # evaluation on a holdout set
        dump(scaler, os.path.join(self.modelpath, 'scaler_' + counter + '.joblib'))
        xtest = self.x[test]
        xtest = scaler.transform(xtest)

        n_test = int(len(ytest) * (1 - VALID_SIZE))

        xvalid = xtest[n_test:]
        yvalid = ytest[n_test:]

        xtest = xtest[:n_test]
        ytest = ytest[:n_test]

        xtrain = np.vstack([xtrain, xvalid])
        ytrain = np.vstack([ytrain, yvalid])

        valid_size = len(xvalid) / len(xtrain)

        trainlogger.debug('the test set has shape %s', xtest.shape)

        ytrain = ytrain.ravel()
        ytest = ytest.ravel()

        optimized_models_split = MLOxidationStates.tune_fit(
            classifiers,
            xtrain,
            ytrain,
            self.max_evals,
            self.timeout,
            self.mix_ratios,
            valid_size,
        )
        res = MLOxidationStates.model_eval(
            optimized_models_split,
            xtrain,
            ytrain,
            xtest,
            ytest,
            counter,
            self.metricspath,
            self.modelpath,
        )
        all_predictions.extend(res)

        # now build an ensemble based on the single models
        ensemble_model, elapsed_time = MLOxidationStates.train_ensemble(
            optimized_models_split,
            xtrain,
            ytrain,
            voting=self.voting,
            calibrate=self.calibrate,
            valid_size=valid_size,
        )
        ensemble_predictions = MLOxidationStates.model_eval(
            [('ensemble', ensemble_model)],
            xtrain,
            ytrain,
            xtest,
            ytest,
            counter,
            self.metricspath,
            self.modelpath,
        )
        all_predictions.extend(ensemble_predictions)
        self.timings.append(elapsed_time)

        return all_predictions

    def track_comet_ml(self):
        """Function to track main parameters and metrics using comet.ml"""
        trainlogger.debug('entering the tracking function')
        experiment = Experiment(
            api_key=os.getenv('COMET_API_KEY', None),
            project_name='mof-oxidation-states',
        )

        mean_time = np.mean(np.array(self.timings))
        self.metrics = MLOxidationStates.summarize_metrics(self.bootstrap_results,
                                                           outpath=self.metricspath,
                                                           timings=mean_time)
        experiment.log_dataset_hash(self.x)
        experiment.log_metrics(self.metrics)
        basemodels = [i for i, _ in classifiers]
        experiment.log_parameter('models', basemodels)
        experiment.log_parameter('n_bootstraps', self.n)
        experiment.log_parameter('max_hyperopt_eval', self.max_evals)
        experiment.log_parameter('timeout_hyperopt', self.timeout)
        experiment.log_parameter('fraction_tpe', self.mix_ratios['tpe'])
        experiment.log_parameter('fraction_random', self.mix_ratios['rand'])
        experiment.log_parameter('fraction_anneal', self.mix_ratios['anneal'])
        experiment.log_parameter('voting', self.voting)
        experiment.log_parameter('size', self.max_size)
        experiment.log_parameter('eval_method', self.eval_method)
        experiment.log_parameter('scaler', self.scalername)
        experiment.log_parameter('calibration_method', self.calibrate)
        experiment.log_parameter('oversampling', self.oversampling)
        experiment.add_tag('initial_model_eval')
        experiment.log_parameter('validation_percentage', VALID_SIZE)
        experiment.log_metric('mean_training_time', mean_time)
        return experiment

    @staticmethod
    def summarize_metrics(metrics: list, outpath: str, timings: float):
        """Calculates summaries of metrics and writes them into .json file for dvc

        Arguments:
            metrics {list} -- list of dictionaries
            outpath {str} -- path to which metrics are writting
            timings {float} -- training time in seconds

        Returns:
            dict -- dictionary with most important metrics
        """
        df = pd.DataFrame(metrics)
        df_ensemble = df[df['model'] == 'ensemble']

        summary_metrics = {
            'mean_accuracy_test': df_ensemble['accuracy_test'].mean(),
            'median_accuracy_test': df_ensemble['accuracy_test'].median(),
            'std_accuracy_test': df_ensemble['accuracy_test'].std(),
            'mean_accuracy_train': df_ensemble['accuracy_train'].mean(),
            'median_accuracy_train': df_ensemble['accuracy_train'].median(),
            'std_accuracy_train': df_ensemble['accuracy_train'].std(),
            'mean_f1_micro_train': df_ensemble['f1_micro_train'].mean(),
            'median_f1_micro_train': df_ensemble['f1_micro_train'].median(),
            'std_f1_micro_train': df_ensemble['f1_micro_train'].std(),
            'mean_f1_micro_test': df_ensemble['f1_micro_test'].mean(),
            'median_f1_micro_test': df_ensemble['f1_micro_test'].median(),
            'std_f1_micro_test': df_ensemble['f1_micro_test'].std(),
            'mean_f1_macro_train': df_ensemble['f1_macro_train'].mean(),
            'median_f1_macro_train': df_ensemble['f1_macro_train'].median(),
            'std_f1_macro_train': df_ensemble['f1_macro_train'].std(),
            'mean_f1_macro_test': df_ensemble['f1_macro_test'].mean(),
            'median_f1_macro_test': df_ensemble['f1_macro_test'].median(),
            'std_f1_macro_test': df_ensemble['f1_macro_test'].std(),
            'mean_precision_train': df_ensemble['precision_train'].mean(),
            'median_precision_train': df_ensemble['precision_train'].median(),
            'std_precision_train': df_ensemble['precision_train'].std(),
            'mean_precision_test': df_ensemble['precision_test'].mean(),
            'median_precision_test': df_ensemble['precision_test'].median(),
            'std_precision_test': df_ensemble['precision_test'].std(),
            'mean_recall_train': df_ensemble['recall_train'].mean(),
            'median_recall_train': df_ensemble['recall_train'].median(),
            'std_recall_train': df_ensemble['recall_train'].std(),
            'mean_recall_test': df_ensemble['recall_train'].mean(),
            'median_recall_test': df_ensemble['recall_train'].median(),
            'std_recall_test': df_ensemble['recall_train'].std(),
            'mean_balanced_accuracy_train': df_ensemble['balanced_accuracy_train'].mean(),
            'median_balanced_accuracy_train': df_ensemble['balanced_accuracy_train'].median(),
            'std_balanced_accuracy_train': df_ensemble['balanced_accuracy_train'].std(),
            'mean_balanced_accuracy_test': df_ensemble['balanced_accuracy_train'].mean(),
            'median_balanced_accuracy_test': df_ensemble['balanced_accuracy_train'].median(),
            'std_balanced_accuracy_test': df_ensemble['balanced_accuracy_train'].std(),
            'mean_training_set_size': df_ensemble['training_points'].mean(),
            'mean_test_set_size': df_ensemble['test_points'].mean(),
            'mean_training_time': timings,
        }

        # now write a .json with metrics for DVC
        with open(os.path.join(outpath, 'train_metrics.json'), 'w') as fp:
            json.dump(summary_metrics, fp)

        return summary_metrics

    def train_test_cv(self):
        """Train an ensemble using a cross-validation technique for evaluation"""
        # Get different sizes for learning curves if needed
        trainlogger.debug('the metrics are saved to %s', self.metricspath)
        trainlogger.debug('the models are saved to %s', self.modelpath)

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

        if self.max_size is not None:
            assert self.max_size <= len(self.y)
            rng = np.random.RandomState(RANDOM_SEED)

            sample_idx = np.arange(self.x.shape[0])
            sampled_idx = rng.choice(sample_idx, size=self.max_size, replace=True)

            self.x = self.x[sampled_idx]
            self.y = self.y[sampled_idx]

        if self.eval_method == 'kfold':
            bs = MLOxidationStates.get_train_test_split(self.x, self.y, self.n)
        elif self.eval_method == 'bootstrap':
            bs = MLOxidationStates.get_bootstrap(self.x, self.y, self.n)
        else:
            bs = MLOxidationStates.get_train_test_split(self.x, self.y, self.n)

        # all_predictions = []
        # do not run this concurrently since the state  of the scaler is not clear!
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for metrics in executor.map(self.train_eval_single, enumerate(list(bs))):
                # all_predictions.extend(predfull)
                self.bootstrap_results.extend(metrics)


@click.command('cli')
@click.argument('xpath')
@click.argument('ypath')
@click.argument('modelpath')
@click.argument('metricspath')
@click.argument('scaler', default='standard')
@click.argument('voting', default='hard')
@click.argument('calibrate', default='spline')
@click.argument('max_size', default=None)
@click.argument('n', default=10)
@click.argument('oversampling', default='smote')
def train_model(
        xpath,
        ypath,
        modelpath,
        metricspath,
        scaler,
        voting,
        calibrate,
        max_size,
        n,
        oversampling,
):
    if not os.path.exists(os.path.abspath(modelpath)):
        os.mkdir(os.path.abspath(modelpath))

    ml_object = MLOxidationStates.from_x_y_paths(
        xpath=os.path.abspath(xpath),
        ypath=os.path.abspath(ypath),
        modelpath=os.path.abspath(modelpath),
        metricspath=os.path.abspath(metricspath),
        scaler=scaler,
        n=int(n),
        voting=voting,
        calibrate=calibrate,
        max_size=int(max_size),
        oversampling=oversampling,
    )
    ml_object.train_test_cv()
    experiment = ml_object.track_comet_ml()
    experiment.log_asset(xpath)
    experiment.log_asset(ypath)


if __name__ == '__main__':
    train_model()  # pylint:disable=no-value-for-parameter
