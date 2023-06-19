"""
The content of this file is based on the scaffold split implementation of DeepChem
https://github.com/deepchem/deepchem

The parts of implementation adapted from DeepChem are shared under the MIT license:

Copyright 2017 PandeLab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AN
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def _generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.

    Args:
        smiles: input SMILES string.
        include_chirality: if True, chirality is included in computed scaffolds.

    Returns:
        Bemis-Murcko scaffold of the input molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n: int = 1000) -> List[List[int]]:
    """
    Returns all scaffolds from the dataset.

    Args:
        dataset: Dataset to be split.
        log_every_n: Controls the logger by dictating how often logger outputs
          will be produced.

    Returns:
        List of indices of each scaffold in the dataset.
    """
    scaffolds = {}
    data_len = len(dataset)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]
    return scaffold_sets


class Splitter:
    """
    Splitters split up Datasets into pieces for training/validation/testing.
    In machine learning applications, it's often necessary to split up a dataset
    into training/validation/test sets. Or to k-fold split a dataset (that is,
    divide into k equal subsets) for cross-validation. The `Splitter` class is
    an abstract superclass for all splitters that captures the common API across
    splitter classes.

    Note that `Splitter` is an abstract superclass. You won't want to
    instantiate this class directly. Rather you will want to use a concrete
    subclass for your application.
    """

    def train_valid_test_split(
        self,
        dataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Splits self into train/validation/test sets.

        Args:
            dataset: Dataset to be split.
            frac_train: The fraction of data to be used for the training split.
            frac_valid: The fraction of data to be used for the validation split.
            frac_test: The fraction of data to be used for the test split.
            seed: Random seed to use.

        Returns:
            A tuple of train, valid and test indices.
        """
        print("Computing train/valid/test indices")
        train_inds, valid_inds, test_inds = self.split(
            dataset,
            frac_train=frac_train,
            frac_test=frac_test,
            frac_valid=frac_valid,
            seed=seed,
        )

        return train_inds, valid_inds, test_inds


class ScaffoldSplitter(Splitter):
    """
    Class for doing data splits based on the scaffold of small molecules.
    """

    def split(
        self,
        dataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        random: bool = True,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits internal compounds into train/validation/test by scaffold.

        Args:
            dataset: Dataset to be split.
            frac_train: The fraction of data to be used for the training split.
            frac_valid: The fraction of data to be used for the validation split.
            frac_test: The fraction of data to be used for the test split.
            random: Creates a random split. If False, the scaffolds are sorted
              by their size and split fully deterministically.

        Returns:
            A tuple of train indices, valid indices, and test indices.
            Each indices is a list of integers.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        scaffold_sets = generate_scaffolds(dataset)
        if random:
            np.random.shuffle(scaffold_sets)

        train_cutoff = frac_train * len(dataset)
        valid_cutoff = (frac_train + frac_valid) * len(dataset)
        train_inds: List[int] = []
        valid_inds: List[int] = []
        test_inds: List[int] = []

        print("About to sort in scaffold sets")
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set
        return train_inds, valid_inds, test_inds
