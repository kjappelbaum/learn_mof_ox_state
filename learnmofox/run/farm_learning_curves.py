# -*- coding: utf-8 -*-
# pylint:disable=too-many-nested-blocks
from __future__ import absolute_import

import os
import subprocess
import sys
from glob import glob
from pathlib import Path

import click

from learnmofox import utils
from learnmofox.utils import SUBMISSION_TEMPLATE, make_if_not_exist

sys.modules["utils"] = utils

to_analyze = [
    # "metal_center_feat",
    # "geometry_feat",
    # "chemistry_feat",
    # "metal_center_chemistry_feat",
    # "metal_center_geometry_feat",
    # "chemistry_geometry_feat",
    # "metal_center_chemistry_geometry_feat",
    # "random_feat",
    # "racs",
    # "racs_metal_center",
    # "racs_geometry",
    # "racs_chemistry",
    # "racs_chemistry_metal_center",
    # "racs_chemistry_metal_center_geometry",
    "chemistry_metal_center_geometry_tight",
]

POINTS = [47000, 30000, 20000, 10000, 5000, 1000, 500, 200, 100]


def check_if_model_exists(modelpath):
    models = glob(os.path.join(modelpath, "*.joblib"))
    if len(models) >= 6:
        return True

    return False


def write_run_command(  # pylint:disable=too-many-function-args, too-many-arguments
    modelpath, datapath, validpath, holdoutpath, outpath, numpoints
) -> str:
    xtrainpath = os.path.join(datapath, "features", "features.npy")
    ytrainpath = os.path.join(datapath, "labels", "labels.npy")
    xvalidpath = os.path.join(validpath, "features.npy")
    yvalidpath = os.path.join(validpath, "labels.npy")

    xtestpath = os.path.join(holdoutpath, "features.npy")
    ytestpath = os.path.join(holdoutpath, "labels.npy")

    bootstraps = 200
    return f"python machine_learn_oxstates/bootstrapped_metrics_cli.py {modelpath} {xtrainpath} {ytrainpath} {xvalidpath} {yvalidpath} {xtestpath} {ytestpath} {outpath} {numpoints} {bootstraps}"  # pylint:disable=line-too-long


def write_slurmfile(name: str, command: str):

    data = {"name": name, "command": command, "time": 72}

    template = SUBMISSION_TEMPLATE.format(**data)
    with open("submit_{}.slurm".format(name), "w") as fh:
        fh.write(template)


def underscore_join(l):
    return "_".join(l)


@click.command("cli")
@click.option("--submit", is_flag=True)
def main(submit):
    make_if_not_exist("learning_curves")
    for estimator in to_analyze:
        modelpath = underscore_join(["model", estimator])
        if check_if_model_exists(modelpath):
            for model in glob(os.path.join(modelpath, "*.joblib")):
                if "ensemble" in model:
                    p = Path(model)
                    modelbasename = os.path.join(
                        "learning_curves", underscore_join([estimator, p.stem])
                    )
                    make_if_not_exist(modelbasename)
                    for point in POINTS:
                        modelpointname = os.path.join(modelbasename, str(point))
                        make_if_not_exist(modelpointname)
                        command = write_run_command(
                            model,
                            underscore_join(["data", estimator]),
                            os.path.join(
                                underscore_join(["houldout", estimator]), "valid"
                            ),
                            underscore_join(["houldout", estimator]),
                            modelpointname,
                            point,
                        )
                        submission_name = underscore_join(
                            [estimator, Path(modelpointname).stem]
                        )
                        write_slurmfile(submission_name, command)

                        if submit:
                            subprocess.call(
                                "sbatch submit_{}.slurm".format(submission_name),
                                shell=True,
                            )


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
