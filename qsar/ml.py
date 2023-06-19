import os
import pickle
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import sklearn.base
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC, SVR
from skorch import NeuralNetClassifier
from skorch.regressor import NeuralNetRegressor
from torch import nn


def train_rf(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    save_name: Optional[str] = None,
    task_type: str = "regression",
    grid: Optional[dict] = None,
) -> Tuple[float, float, dict]:
    """
    Runs a hyperparameter search for the random forest algorithm.

    Args:
        X_train: training data features.
        X_valid: validation data features.
        X_test: testing data features.
        y_train: training data labels.
        y_valid: validation data labels.
        y_test: testing data labels.
        save_name: name of the file where the results are saved; if None, nothing is saved.
        task_type: type of the learning problem (regression or classification).
        grid: dictionary of hyperparameters to tune; if None, a default hyperparameter set is used.

    Returns:
        (valid score, test score, best hyperparameters)
    """
    if not grid:
        grid = {"n_estimators": [300]}
    if task_type == "regression":
        model_fn = lambda *args, **kwargs: RandomForestRegressor(
            *args, **kwargs, n_jobs=-1
        )
    elif task_type == "classification":
        model_fn = lambda *args, **kwargs: RandomForestClassifier(
            *args, **kwargs, n_jobs=-1
        )
    else:
        raise ValueError(
            'Incorrect task type, must be either "regression" or "classification".'
        )
    return run_grid_search(
        model_fn, grid, save_name, X_train, y_train, X_valid, y_valid, X_test, y_test
    )


def train_svm(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    save_name: Optional[str] = None,
    task_type: str = "regression",
    grid: Optional[dict] = None,
) -> Tuple[float, float, dict]:
    """
    Runs a hyperparameter search for the SVM algorithm.

    Args:
        X_train: training data features.
        X_valid: validation data features.
        X_test: testing data features.
        y_train: training data labels.
        y_valid: validation data labels.
        y_test: testing data labels.
        save_name: name of the file where the results are saved; if None, nothing is saved.
        task_type: type of the learning problem (regression or classification).
        grid: dictionary of hyperparameters to tune; if None, a default hyperparameter set is used.

    Returns:
        (valid score, test score, best hyperparameters)
    """
    if not grid:
        grid = {
            "kernel": ["rbf"],
            "C": [2**i for i in range(7)],
        }
    if task_type == "regression":
        model_fn = lambda *args, **kwargs: SVR(*args, **kwargs)
    elif task_type == "classification":
        model_fn = lambda *args, **kwargs: SVC(*args, **kwargs, probability=True)
    else:
        raise ValueError(
            'Incorrect task type, must be either "regression" or "classification".'
        )
    return run_grid_search(
        model_fn, grid, save_name, X_train, y_train, X_valid, y_valid, X_test, y_test
    )


def make_network(
    hidden_size: float,
    input_size: float,
    num_layers: int,
    batch_norm: bool,
    dropout: float,
    activation: torch.nn.Module,
) -> torch.Module:
    """
    Builds a neural network architecture using the given hyperparameters.

    Args:
        hidden_size: size of the hidden dimension.
        input_size: size of the input features.
        num_layers: number of layers.
        batch_norm: if True, batch norm is used.
        dropout: dropout rate.
        activation: activation function.

    Returns:
        Neural network architecture.
    """
    layers = [torch.nn.Linear(input_size, hidden_size)]
    if batch_norm:
        layers.append(torch.nn.BatchNorm1d(hidden_size))
    if dropout:
        layers.append(torch.nn.Dropout(dropout))
    layers.append(activation())
    for _ in range(num_layers):
        layers.append(torch.nn.Linear(hidden_size, hidden_size))
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(hidden_size))
        if dropout:
            layers.append(torch.nn.Dropout(dropout))
        layers.append(activation())
    layers.append(torch.nn.Linear(hidden_size, 1))
    net = torch.nn.Sequential(*layers)
    return net


def train_nn(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    save_name: Optional[str] = None,
    task_type: str = "regression",
    grid: Optional[dict] = None,
) -> Tuple[float, float, dict]:
    """
    Runs a hyperparameter search for the artificial neural network.

    Args:
        X_train: training data features.
        X_valid: validation data features.
        X_test: testing data features.
        y_train: training data labels.
        y_valid: validation data labels.
        y_test: testing data labels.
        save_name: name of the file where the results are saved; if None, nothing is saved.
        task_type: type of the learning problem (regression or classification).
        grid: dictionary of hyperparameters to tune; if None, a default hyperparameter set is used.

    Returns:
        (valid score, test score, best hyperparameters)
    """
    input_size = X_train.shape[1]
    X_train = X_train.to_numpy().astype(np.float32)
    X_valid = X_valid.to_numpy().astype(np.float32)
    X_test = X_test.to_numpy().astype(np.float32)
    y_train = y_train.to_numpy().reshape(-1, 1).astype(np.float32)
    y_valid = y_valid.to_numpy().reshape(-1, 1).astype(np.float32)
    y_test = y_test.to_numpy().reshape(-1, 1).astype(np.float32)
    if not grid:
        grid = {
            "hidden_size": [1024, 512, 256],
            "lr": [0.0001, 0.001],
            "num_layers": [2, 3, 4, 5],
            "batch_norm": [True],
            "dropout": [0.2, 0.5],
            "activation": [nn.ReLU],
            "epochs": [500],
            "batch_size": [64, 128, 256],
            "optimizer": [torch.optim.Adam],
        }
    if task_type == "regression":
        model_fn = (
            lambda batch_size, epochs, lr, optimizer, **kwargs: NeuralNetRegressor(
                make_network(input_size=input_size, **kwargs),
                max_epochs=epochs,
                batch_size=batch_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
                lr=lr,
                optimizer=optimizer,
                train_split=None,
                verbose=False,
            )
        )
    elif task_type == "classification":
        model_fn = (
            lambda batch_size, epochs, lr, optimizer, **kwargs: NeuralNetClassifier(
                make_network(input_size=input_size, **kwargs),
                max_epochs=epochs,
                batch_size=batch_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
                lr=lr,
                optimizer=optimizer,
                train_split=None,
                verbose=False,
                criterion=torch.nn.BCEWithLogitsLoss,
            )
        )
    else:
        raise ValueError(
            'Incorrect task type, must be either "regression" or "classification".'
        )
    return run_grid_search(
        model_fn, grid, save_name, X_train, y_train, X_valid, y_valid, X_test, y_test
    )


def run_grid_search(
    model_fn: sklearn.base.BaseEstimator,
    grid: Optional[dict],
    save_name: Optional[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Runs a hyperparameter search for the given model.

    Args:
        model_fn: machine learning model constructor.
        grid: dictionary of hyperparameters to tune; if None, a default hyperparameter set is used.
        X_train: training data features.
        X_valid: validation data features.
        X_test: testing data features.
        y_train: training data labels.
        y_valid: validation data labels.
        y_test: testing data labels.

    Returns:
        (valid score, test score, best hyperparameters)
    """
    hps_results = []
    best_score = -np.inf
    best_params = None

    for hyperparameters in ParameterGrid(grid):
        model = model_fn(**hyperparameters)
        model.fit(X_train, y_train)
        score = model.score(X_valid, y_valid)

        hps_results.append({f"val_score": score, **hyperparameters})
        if save_name:
            pd.DataFrame(hps_results).to_csv(save_name + "_grid.csv")

        if score > best_score:
            if save_name:
                with open(save_name + ".p", "wb") as file:
                    pickle.dump(model, file)
                test_preds = model.predict(X_test)
                pd.DataFrame(
                    data={"preds": test_preds.flatten(), "true": y_test.flatten()}
                ).to_csv(save_name + "_preds.csv")
            best_score = score
            best_test_score = model.score(X_test, y_test)
            best_params = hyperparameters
    score_valid, score_test = best_score, best_test_score
    parameters = best_params
    return score_valid, score_test, parameters


def train_model(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    model_name: str,
    descriptors: Optional[list] = None,
    save_path: str = "./models",
    task_type: str = "regression",
):
    """
    Trains a machine learning model and returns its best validation and test scores
    along with the best hyperparameter set found in a grid search.

    Args:
        df_train: training data frame (must contain smiles and y).
        df_valid: validation data frame (must contain smiles and y).
        df_test: testing data frame (must contain smiles and y).
        model_name: name of the machine learning algorithm (supported: rf, svm, nn).
        descriptors: list of the descriptors to us; if None, all computed descriptors are used.
        save_path: path to where results should be saved.
        task_type: type of the learning problem (regression or classification).

    Returns:
        (valid score, test score, best hyperparameters)
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    X_train = np.nan_to_num(df_train.loc[:, descriptors].astype(np.float32))
    X_valid = np.nan_to_num(df_valid.loc[:, descriptors].astype(np.float32))
    X_test = np.nan_to_num(df_test.loc[:, descriptors].astype(np.float32))

    y_train = np.nan_to_num(df_train["y"].astype(np.float32))
    y_valid = np.nan_to_num(df_valid["y"].astype(np.float32))
    y_test = np.nan_to_num(df_test["y"].astype(np.float32))

    if model_name == "rf":
        train_model = train_rf
    elif model_name == "svm":
        train_model = train_svm
    elif model_name == "nn":
        train_model = train_nn
    else:
        raise ValueError("Incorrect model type.")

    save_name = os.path.join(save_path, f"{now_str}-{model_name}")
    score_valid, score_test, parameters = train_model(
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        save_name=save_name,
        task_type=task_type,
    )

    return score_valid, score_test, parameters
