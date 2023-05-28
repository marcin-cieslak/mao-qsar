import pytest

from qsar.data import load_data
from qsar.fingerprints import calculate_fingerprints


@pytest.mark.parametrize("length", [512, 1024, 2048])
def test_morgan(length):
    dataset = load_data("./test_data.csv", "docking_score")
    dataset = calculate_fingerprints(dataset, "morgan", length)
    assert "smiles" in dataset.columns
    assert "y" in dataset.columns
    assert dataset.shape[1] == length + 2


@pytest.mark.parametrize("length", [512, 1024, 2048])
def test_avalon(length):
    dataset = load_data("./test_data.csv", "docking_score")
    dataset = calculate_fingerprints(dataset, "avalon", length)
    assert "smiles" in dataset.columns
    assert "y" in dataset.columns
    assert dataset.shape[1] == length + 2


def test_maccs():
    dataset = load_data("./test_data.csv", "docking_score")
    dataset = calculate_fingerprints(dataset, "maccs")
    assert "smiles" in dataset.columns
    assert "y" in dataset.columns
    assert dataset.shape[1] == 167 + 2


def test_atompair():
    dataset = load_data("./test_data.csv", "docking_score")
    dataset = calculate_fingerprints(dataset, "atompair")
    assert "smiles" in dataset.columns
    assert "y" in dataset.columns
    assert dataset.shape[1] == 51 + 2
