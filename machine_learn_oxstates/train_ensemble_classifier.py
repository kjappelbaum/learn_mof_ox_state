# -*- coding: utf-8 -*-
# pylint:disable=too-many-arguments, too-many-locals, line-too-long, logging-fstring-interpolation
"""
Trains an ensemble classifier to predict the oxidation state
Produces a  outpath/train_metrics.json file for DVC
"""
from __future__ import absolute_import
from functools import partial
import time
import numpy as np
import os
import json
import pickle
import logging
from typing import Tuple
from hyperopt import tpe, anneal, rand, mix
from hpsklearn.estimator import hyperopt_estimator
from hpsklearn import components
from mlxtend.evaluate import BootstrapOutOfBag
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import pandas as pd
from joblib import dump
import concurrent.futures
import click
from comet_ml import Experiment

COUNTER = 0
RANDOM_SEED = 1234
STARTTIMESTRING = time.strftime('%Y%m%d-%H%M%S')

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
            max_evals: int = 500,
            voting: str = 'soft',
            timeout: int = 600,
            max_workers: int = 4,
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
            max_size: int,
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
            modelpath=modelpath,
            metricspath=metricspath,
        )

    @staticmethod
    def train_ensemble(models: list, X: np.array, y: np.array, voting='soft') -> Tuple[CalibratedClassifierCV, float]:
        """Collects base models into a voting classifier, trains it and then performs
        probability calibration

        Arguments:
            models {list} -- list of optimized base models
            X {np.array} -- feature matrix
            y {np.array} -- label vector

        Keyword Arguments:
            voting {str} -- voting mechanism (hard or soft) (default: {"soft"})

        Returns:
            [CalibratedClassifierCV, float] -- [description]
        """
        trainlogger.debug('training ensemble model')
        vc = VotingClassifier(models, voting=voting)
        startime = time.process_time()
        vc.train(X, y)
        endtime = time.process_time()
        elapsed_time = startime - endtime
        isotonic = CalibratedClassifierCV(vc, cv=10, method='isotonic')

        return isotonic, elapsed_time

    @staticmethod
    def tune_fit(
            models: list,
            X_valid: np.ndarray,
            y_valid: np.ndarray,
            max_evals: int = 500,
            timeout: int = 10 * 60,
            mix_ratios: dict = {
                'rand': 0.1,
                'tpe': 0.8,
                'anneal': 0.1,
            },  # pylint:disable=dangerous-default-value
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
            )

            m.fit(X_valid, y_valid)

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
            postfix: str = COUNTER,
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

            auc_train = auc(train_true, train_predict)
            auc_test = auc(test_true, test_predict)
            balanced_accuracy_train = balanced_accuracy_score(train_true, train_predict)
            balanced_accuracy_test = balanced_accuracy_score(test_true, test_predict)
            precision_train = precision_score(train_true, train_predict)
            precision_test = precision_score(train_true, train_predict)
            recall_train = recall_score(train_true, train_predict)
            recall_test = recall_score(test_true, test_predict)

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
                'auc_train': auc_train,
                'auc_test': auc_test,
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

        return arrays, prediction

    def train_eval_single(self, tt_indices):
        """Peforms a optimize, train, evaluation loop on one fold

        Arguments:
            tt_indices {tuple} -- indices for training and test set

        Returns:
            list -- list of dictionaries of model performance metrics
        """

        trainlogger.debug('entered the function that trains one fold')
        all_predictions = []
        counter = str(COUNTER)
        train, test = tt_indices

        scaler = self.scaler

        xtrain = scaler.fit_transform(self.x[train])
        # save the latest scaler so we can use it later with latest model for
        # evaluation on a holdout set

        dump(scaler, os.path.join(self.modelpath, 'scaler_' + counter))
        xtest = scaler.transform(self.x[test])

        optimized_models_split = MLOxidationStates.tune_fit(
            classifiers,
            xtrain,
            self.y[train],
            self.max_evals,
            self.timeout,
            self.mix_ratios,
        )
        res = MLOxidationStates.model_eval(
            optimized_models_split,
            xtrain,
            self.y[train],
            xtest,
            self.y[test],
            counter,
            self.metricspath,
            self.modelpath,
        )
        all_predictions.extend(res)
        ensemble_model, elapsed_time = MLOxidationStates.train_ensemble(optimized_models_split,
                                                                        self.x[train],
                                                                        self.y[train],
                                                                        voting=self.voting)
        ensemble_predictions = MLOxidationStates.model_eval([('ensemble', ensemble_model)], self.x, self.y, train, test,
                                                            counter)
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
        experiment.add_tag('initial_test')
        experiment.log_metric('mean_training_time', mean_time)

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
            'mean_auc_train': df_ensemble['auc_train'].mean(),
            'median_auc_train': df_ensemble['auc_train'].median(),
            'std_auc_train': df_ensemble['auc_train'].std(),
            'mean_auc_test': df_ensemble['auc_test'].mean(),
            'median_auc_test': df_ensemble['auc_test'].median(),
            'std_auc_test': df_ensemble['auc_test'].std(),
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
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            for _, metrics in executor.map(self.train_eval_single, bs):
                COUNTER += 1
                # all_predictions.extend(predfull)
                self.bootstrap_results.append(metrics)


@click.command('cli')
@click.argument('xpath')
@click.argument('ypath')
@click.argument('modelpath')
@click.argument('metricspath')
@click.argument('scaler', default='standard')
@click.argument('voting', default='soft')
@click.argument('max_size', default=None)
@click.argument('n', default=10)
def train_model(xpath, ypath, modelpath, metricspath, scaler, voting, max_size, n):
    if not os.path.exists(os.path.abspath(modelpath)):
        os.mkdir(os.path.abspath(modelpath))

    ml_object = MLOxidationStates.from_x_y_paths(
        os.path.abspath(xpath),
        os.path.abspath(ypath),
        os.path.abspath(modelpath),
        os.path.abspath(metricspath),
        scaler,
        int(n),
        voting,
        int(max_size),
    )
    ml_object.train_test_cv()


if __name__ == '__main__':
    train_model()  # pylint:disable=no-value-for-parameter
