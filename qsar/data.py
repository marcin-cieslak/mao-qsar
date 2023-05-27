from typing import Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem

from qsar.splitters.scaffold import ScaffoldSplitter


def load_data(
    data_path: str, activity_column_name: str, sep: str = ";"
) -> pd.DataFrame:
    """
    Loads activity data from a CSV file.

    Args:
        data_path: path to the input CSV file.
        activity_column_name: name of the activity column.
        sep: separator used in the CSV file.

    Returns:
        Dataframe containing activity data loaded into two columns: "smiles" and "y".
    """
    if ".csv" not in data_path:
        data_path += ".csv"
    df = pd.read_csv(data_path, sep=sep)
    df.drop_duplicates(subset=["smiles"], keep="first", inplace=True)

    labels = []
    smileses = []
    for i, row in df.iterrows():
        s = row.smiles
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            labels.append(row[activity_column_name])
            smileses.append(s)
    df = pd.DataFrame(data={"y": labels, "smiles": smileses})

    # correct labels if loaded as strings using commas instead of dots
    if pd.api.types.is_string_dtype(df.y.dtype):
        df.y = df.y.str.replace(",", ".").astype(float)

    return df


def split_data(
    df: pd.DataFrame,
    method: str = "random",
    test_size: float = 0.1,
    valid_size: float = 0.1,
    return_idx: bool = False,
) -> Tuple[
    Union[pd.DataFrame, np.ndarray],
    Union[pd.DataFrame, np.ndarray],
    Union[pd.DataFrame, np.ndarray],
]:
    """
    Splits the data in the dataframe into three subsets: train, valid, test.

    Args:
        df: DataFrame with your data
        method: method of the split; it can be either `random` or `scaffold`
        test_size: the fraction of data to be used for the testing split.
        valid_size: the fraction of data to be used for the validation split.
        return_idx: if True, only indices of subsets are returned instead of dataframes.

    Returns:
        (train df, valid df, test df)
    """
    if method == "random":
        indices = np.random.permutation(len(df))
        train_idx = indices[: int(len(df) * (1 - test_size - valid_size))]
        valid_idx = indices[
            int(len(df) * (1 - test_size - valid_size)) : int(len(df) * (1 - test_size))
        ]
        test_idx = indices[int(len(df) * (1 - test_size)) :]
    elif method == "scaffold":
        splitter = ScaffoldSplitter()
        train_idx, valid_idx, test_idx = splitter.split(
            df,
            frac_train=1.0 - valid_size - test_size,
            frac_valid=valid_size,
            frac_test=test_size,
        )
    else:
        raise ValueError(
            f"Split `{method}` is not implemented. You can use `random` or `scaffold`."
        )
    if return_idx:
        return train_idx, valid_idx, test_idx
    else:
        df_train, df_valid, df_test = (
            df.iloc[train_idx],
            df.iloc[valid_idx],
            df.iloc[test_idx],
        )
        return df_train, df_valid, df_test


def convert_datawarrior_to_csv(filename: str):
    """
    Converts txt files saved using DataWarrior to comma separated csv files and saves
    it in the same directory.

    Args:
        filename: path to the txt file generated with DataWarrior
    """
    csv_filename = filename.replace(".txt", ".csv")
    df = pd.read_csv(filename, sep="\t")
    df.drop(labels="Structure [idcode]", axis="columns", inplace=True)
    df.to_csv(csv_filename, index=False)
