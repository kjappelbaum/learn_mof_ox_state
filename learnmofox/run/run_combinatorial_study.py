# -*- coding: utf-8 -*-
# pylint:disable=line-too-long
from __future__ import absolute_import
from __future__ import print_function
import os
import click
import subprocess

TRAINING_SET = ['mofs', 'mp_mofs']

METAL_CENTER_FEATURES = [
    'column',
    'row',
    'valenceelectrons',
    'diffto18electrons',
    'sunfilled',
    'punfilled',
    'dunfilled',
]

GEOMETRY_FEATURES = ['crystal_nn_fingerprint', 'behler_parinello']

CHEMISTRY_FEATURES = ['local_property_stats']

FEATURE_SETS = [
    ('metal_center_feat', METAL_CENTER_FEATURES),
    ('geometry_feat', GEOMETRY_FEATURES),
    ('chemistry_feat', CHEMISTRY_FEATURES),
    ('metal_center_chemistry_feat', METAL_CENTER_FEATURES + CHEMISTRY_FEATURES),
    ('metal_center_geometry_feat', METAL_CENTER_FEATURES + GEOMETRY_FEATURES),
    ('chemistry_geometry_feat', CHEMISTRY_FEATURES + GEOMETRY_FEATURES),
    (
        'metal_center_chemistry_geometry_feat',
        METAL_CENTER_FEATURES + CHEMISTRY_FEATURES + GEOMETRY_FEATURES,
    ),
    ('random_feat', ['random_column']),
    ('racs', []),
    ('racs_metal_center', METAL_CENTER_FEATURES),
    ('racs_geometry', GEOMETRY_FEATURES),
    ('racs_chemistry', CHEMISTRY_FEATURES),
    ('racs_chemistry_metal_center', CHEMISTRY_FEATURES + METAL_CENTER_FEATURES),
    (
        'racs_chemistry_metal_center_geometry',
        CHEMISTRY_FEATURES + METAL_CENTER_FEATURES + GEOMETRY_FEATURES,
    ),
    (
        'chemistry_metal_center_geometry_tight',
        CHEMISTRY_FEATURES + METAL_CENTER_FEATURES + ['crystal_nn_no_steinhardt'],
    ),
]

RACSDATAPATH = ('/scratch/kjablonk/oxidationstates/machine_learn_oxstates/data/df_racs_cleaned.csv')
FEATURESPATH = ('/scratch/kjablonk/oxidationstates/machine_learn_oxstates/data/20190928_features')
LABELSPATH = '/scratch/kjablonk/oxidationstates/machine_learn_oxstates/data/labels/20190917_labels.pkl'
"""
For each of the feature sets we will now do full hyperparameter search with hyperopt and many cycles on a smaller training set.
Then we use this to run the full training and the calibration and follow this up by a small testing.

This is the runscript for the first part in which we do the hyperparameter optimization for each feature set.
"""

SUBMISSION_TEMPLATE = """#!/bin/bash -l
#SBATCH --chdir ./
#SBATCH --mem 26GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name {name}
#SBATCH --time {time}:00:00
#SBATCH --partition serial

source /home/kjablonk/anaconda3/bin/activate
conda activate ml
export COMET_API_KEY='Nqp9NvaVztUCG2exYT9vV2Dl0'

{command}
"""


def write_slurmfile(name: str, command: str, runtype: str):
    if runtype == 'feat':
        time = 72
    else:
        time = 72

    data = {'name': name, 'command': command, 'time': time}

    template = SUBMISSION_TEMPLATE.format(**data)
    slurmname = 'submit_{}_{}.slurm'.format(name, runtype)
    with open(slurmname, 'w') as fh:
        fh.write(template)

    return slurmname


def _make_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def write_featureselelect_command(name: str, features: list) -> str:
    datapath = '_'.join(['data', name])

    _make_if_not_exists(datapath)

    featureoutspath = os.path.join(datapath, 'features')

    _make_if_not_exists(featureoutspath)

    labelsoutpath = os.path.join(datapath, 'labels')

    _make_if_not_exists(labelsoutpath)

    helperoutpath = os.path.join(datapath, 'helper')

    _make_if_not_exists(helperoutpath)

    holdoutpath = '_'.join(['houldout', name])

    _make_if_not_exists(holdoutpath)

    if name == 'racs':  # pylint:disable=no-else-return
        return f'run_featurecollection --only_racs {FEATURESPATH}  {LABELSPATH} {labelsoutpath} {featureoutspath} {helperoutpath} 0.2 {holdoutpath} 60000 {RACSDATAPATH} column row crystal_nn_no_steinhardt'
    elif name == 'metal':
        features = ' '.join(features)
        return f'run_featurecollection --do_not_drop_duplicates {FEATURESPATH} {LABELSPATH} {labelsoutpath} {featureoutspath} {helperoutpath} 0.2 {holdoutpath} 60000 None {features}'
    elif 'racs_' in name:
        features = ' '.join(features)
        return f'run_featurecollection {FEATURESPATH} {LABELSPATH} {labelsoutpath} {featureoutspath} {helperoutpath} 0.2 {holdoutpath} 60000  {RACSDATAPATH} {features}'
    else:
        features = ' '.join(features)
        return f'run_featurecollection {FEATURESPATH} {LABELSPATH} {labelsoutpath} {featureoutspath} {helperoutpath} 0.2 {holdoutpath} 60000 None {features}'


def write_run_command(name: str) -> str:
    datapath = '_'.join(['data', name])
    featureoutspath = os.path.join(datapath, 'features', 'features.npy')
    labelsoutpath = os.path.join(datapath, 'labels', 'labels.npy')
    metricsoutpath = os.path.join(datapath, 'metrics')
    _make_if_not_exists(metricsoutpath)
    modelpath = '_'.join(['model', name])
    _make_if_not_exists(modelpath)

    return f'python machine_learn_oxstates/learnmofox/train_ensemble_classifier.py {featureoutspath} {labelsoutpath} {modelpath} {metricsoutpath} standard soft isotonic 40000 20 none --train_one_fold'


@click.command('cli')
@click.option('--create_featuresets', is_flag=True)
@click.option('--run_model_selection', is_flag=True)
@click.option('--submit', is_flag=True)
def main(create_featuresets, run_model_selection, submit):
    for name, featureset in FEATURE_SETS:
        if create_featuresets:
            runtype = 'feat'
            command = write_featureselelect_command(name, featureset)
        elif run_model_selection:
            runtype = 'train'
            command = write_run_command(name)

        print(f'writing slurmfile for {name}')
        slurmname = write_slurmfile(name, command, runtype)

        if submit:
            subprocess.call('sbatch {}'.format(slurmname), shell=True)


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
