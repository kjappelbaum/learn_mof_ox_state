# -*- coding: utf-8 -*-
"""
This module takes a trained voting classifier and trains it.
Usually, this will be used to train on a larger dataset, for which not the
expensive hyperopt search should be performed.

This assumed that we read a VotingClassifier, which has the private _fit and _calibrate
methods.
"""
import os
import time

import click
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class TrainVotingClassifier:

    def __init__(  # pylint:disable=too-many-arguments
        self,
        votingclassifier,
        X,
        y,
        calibration=None,
        voting=None,
        outdir=None,
        scaler='minmax',
        validX=None,
        validy=None,
    ):
        self.votingclassifier = votingclassifier
        self.X = X
        self.y = y
        self.validX = validX
        self.validy = validy
        self.y = self.y.astype(np.int)
        self.validy = self.validy.astype(np.int)
        # print(
        #     f'the class labels in the validation data are {np.unique(self.validy)}, the ones in the training set are {np.unique(self.y)}'
        # )
        assert len(np.unique(self.validy)) <= len(np.unique(self.y))

        assert len(self.X) == len(self.y)

        self.calibration = calibration
        self.voting = voting
        self.votingclassifier.voting = self.voting
        self.outdir = outdir
        if self.outdir is None:
            self.outdir = os.getcwd()
        self.traintime = None

        if scaler == 'robust':
            self.scalername = 'robust'
            self.scaler = RobustScaler()
        elif scaler == 'standard':
            self.scalername = 'standard'
            self.scaler = StandardScaler()
        elif scaler == 'minmax':
            self.scalername = 'minmax'
            self.scaler = MinMaxScaler()

        # initialize voting and calibration if None based on the settings of the classifier
        if self.calibration is None:
            self.calibration = self.votingclassifier.voting

        if self.voting is None:
            self.voting = self.votingclassifier.voting

    @classmethod
    def from_files(  # pylint:disable=too-many-arguments, too-many-locals
        cls,
        joblibpath,
        featurematrixpath,
        labelpath,
        calibration=None,
        voting=None,
        outdir=None,
        scaler='standard',
        validX=None,
        validy=None,
    ):
        model = load(joblibpath)
        X = np.load(featurematrixpath)
        y = np.load(labelpath)
        validXdata = np.load(validX)
        validydata = np.load(validy)

        # print(
        #     f'the class labels in the validation data are {np.unique(validydata)}, the ones in the training set are {np.unique(y)}'
        # )
        return cls(model, X, y, calibration, voting, outdir, scaler, validXdata, validydata)

    def _fit(self):
        self.X = self.scaler.fit_transform(self.X)
        self.validX = self.scaler.transform(self.validX)

        startime = time.process_time()
        self.votingclassifier._fit(self.X, self.y)  # pylint:disable=protected-access

        # use validation set to do probability calibration
        # print(f'sent {np.unique(self.validy)} to calibration')
        self.votingclassifier._calibrate_base_estimators(  # pylint:disable=protected-access
            self.calibration, self.validX, self.validy)
        endtime = time.process_time()
        self.traintime = endtime - startime

    def _dump(self):
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        self.votingclassifier._check_is_fitted()  # pylint:disable=protected-access
        dump(self.votingclassifier, os.path.join(self.outdir, 'votingclassifier.joblib'))
        dump(self.scaler, os.path.join(self.outdir, 'scaler.joblib'))

    def return_models(self):
        return self.votingclassifier, self.scaler

    def train(self):
        self._fit()
        self._dump()


@click.command('cli')
@click.argument('modelpath')
@click.argument('featurepath')
@click.argument('labelpath')
@click.argument('calibration')
@click.argument('voting')
@click.argument('outdir')
@click.argument('scaler')
@click.argument('validx')
@click.argument('validy')
def main(  # pylint:disable=too-many-arguments
    modelpath,
    featurepath,
    labelpath,
    calibration,
    voting,
    outdir,
    scaler,
    validx,
    validy,
):

    vc = TrainVotingClassifier.from_files(
        modelpath,
        featurepath,
        labelpath,
        calibration,
        voting,
        outdir,
        scaler,
        validx,
        validy,
    )
    vc.train()


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
