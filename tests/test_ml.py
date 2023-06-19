import torch

from qsar.data import load_data, split_data
from qsar.fingerprints import calculate_fingerprints
from qsar.ml import train_nn, train_rf


def test_rf_regression():
    dataset = load_data("./test_data.csv", "docking_score")
    dataset = calculate_fingerprints(dataset, "morgan")
    data_train, data_valid, data_test = split_data(dataset)
    desc_cols = [column not in ("smiles", "y") for column in data_train.columns]
    score_valid, score_test, parameters = train_rf(
        data_train.iloc[:, desc_cols],
        data_valid.iloc[:, desc_cols],
        data_test.iloc[:, desc_cols],
        data_train["y"],
        data_valid["y"],
        data_test["y"],
    )
    assert score_valid <= 1.0
    assert score_test <= 1.0
    assert isinstance(parameters, dict)


def test_rf_classification():
    dataset = load_data("./test_data_classification.csv", "docking_score")
    dataset = calculate_fingerprints(dataset, "morgan")
    data_train, data_valid, data_test = split_data(dataset)
    desc_cols = [column not in ("smiles", "y") for column in data_train.columns]
    score_valid, score_test, parameters = train_rf(
        data_train.iloc[:, desc_cols],
        data_valid.iloc[:, desc_cols],
        data_test.iloc[:, desc_cols],
        data_train["y"],
        data_valid["y"],
        data_test["y"],
        task_type="classification",
    )
    assert score_valid <= 1.0
    assert score_test <= 1.0
    assert isinstance(parameters, dict)


def test_nn_regression():
    dataset = load_data("./test_data.csv", "docking_score")
    dataset = calculate_fingerprints(dataset, "morgan")
    data_train, data_valid, data_test = split_data(dataset)
    desc_cols = [column not in ("smiles", "y") for column in data_train.columns]
    score_valid, score_test, parameters = train_nn(
        data_train.iloc[:, desc_cols],
        data_valid.iloc[:, desc_cols],
        data_test.iloc[:, desc_cols],
        data_train["y"],
        data_valid["y"],
        data_test["y"],
        grid={
            "hidden_size": [256],
            "lr": [0.0001],
            "num_layers": [2],
            "batch_norm": [True],
            "dropout": [0.0],
            "activation": [torch.nn.ReLU],
            "epochs": [1],
            "batch_size": [64],
            "optimizer": [torch.optim.Adam],
        },
    )
    assert score_valid <= 1.0
    assert score_test <= 1.0
    assert isinstance(parameters, dict)
