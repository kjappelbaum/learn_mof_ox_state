# -*- coding: utf-8 -*-
"""
This module performs inference on new structures
"""
from __future__ import absolute_import
from pymatgen import Structure
#from mine_mof_oxstate.featurize import (
#    GetFeatures,)  # ToDo: maybe migrate module this into this package
from joblib import load

TRAINED_METALS = []  # we will do inference for those


class InferOxStates:

    def __init__(self, structure: Structure, modelpath: str = None, scalerpath: str = None):
        self.structure = structure
        self.TRAINED_METALS = TRAINED_METALS
        self.moddel = load(modelpath)
        self.scaler = load(scalerpath)
        self.predictions = None

    @classmethod
    def from_cif(cls, cifpath: str):
        s = Structure.from_file(cifpath)
        return cls(s)

    def _featurize(self):
        ...

    def _predict(self):
        ...

    def run_inference(self):
        return self.predictions

    @property
    def predictions(self):
        return self.predictions

    def _print_cif(self):
        if self.predictions is not None:
            pass
        else:
            _ = self.run_inference()

        # then set property in cif
