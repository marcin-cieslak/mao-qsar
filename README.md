# MAO QSAR

This repository contains data and code that can
be used to train machine learning model for the
MAO-A and MAO-B QSAR study. This study is based
on activity data downloaded from ChEMBL and the
results of molecular docking (using Smina). The
prepared datasets for both targets and
pre-computed docking scores can be found in the
`data` folder. The code for training QSAR
models is included in the `qsar` directory.

## Conda Environment

Install our conda environment by using the
following command inside the repo directory:

```bash
conda env create -f environment.yml
```

## Example Code

To train machine learning models on the
datasets pulled from ChEMBL (reported activity
or pre-computed docking scores), adapt the
code snippet below:

```python
from qsar.data import load_data, split_data
from qsar.fingerprints import calculate_fingerprints
from qsar.ml import train_rf


dataset = load_data(
    "data/mao_a_docking_score.csv",
    "docking_score"
)
dataset = calculate_fingerprints(dataset, "morgan")
data_train, data_valid, data_test = split_data(
    dataset,
    method='random'
)
desc_cols = [
    column not in ("smiles", "y")
    for column in data_train.columns
]
score_valid, score_test, parameters = train_rf(
    data_train.iloc[:, desc_cols],
    data_valid.iloc[:, desc_cols],
    data_test.iloc[:, desc_cols],
    data_train["y"],
    data_valid["y"],
    data_test["y"],
)
print(score_test)
```

## KS Data Splitting Method

This repository contains our custom method
for regression data stratification that makes
the distribution of training labels similar
to the distribution of testing labels. This
method is based on the Kolmogorov-Smirnov *D*
statistic and can be combined with the scaffold
splitting method.