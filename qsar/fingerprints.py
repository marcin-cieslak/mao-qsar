import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.rdMolDescriptors import GetAtomPairFingerprint


def calculate_fingerprints(df: pd.DataFrame, kind: str, length: int = 1024):
    """
    Calculates fingerprints for the input compounds.

    Args:
        df: input dataframe containing the "smiles" column.
        kind: type of the fingerprint to be calculated.
        length: length of the fingerprint (if supported).

    Returns:
        Dataframe with the calculated fingerprints.
    """
    if kind.lower() == "morgan":
        calculate_fingerprint = lambda mol: AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=length
        )
    elif kind.lower() == "avalon":
        calculate_fingerprint = lambda mol: pyAvalonTools.GetAvalonFP(mol, nBits=length)
    elif kind.lower() == "maccs":
        calculate_fingerprint = MACCSkeys.GenMACCSKeys
    elif kind.lower() == "atompair":
        calculate_fingerprint = GetAtomPairFingerprint
    else:
        raise ValueError("Fingerprint type not supported.")

    fps = []
    labels = []
    smileses = []
    for i, row in df.iterrows():
        s = row.smiles
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = calculate_fingerprint(mol)
            fp = np.array(fp)
            fps.append(fp)
            labels.append(row["y"])
            smileses.append(s)

    fps = np.stack(fps)
    df = pd.DataFrame(data={"y": labels, "smiles": smileses})

    for i, column in enumerate(fps.T):
        df[f"fp{i}"] = column
    return df
