"""
Module to be used to clean data in lab,
currentily it cannot be used as standalone application, it must be imported,
It cannot plot any data, if requested it can
maybe in the future, depending on how many application we need to use ti can be expanded
"""
# %%
import pandas as pd
from pathlib import Path
import numpy as np

# import os


def clean(
    path_data: str | Path,
    path_zero: str | Path,
    params_data: dict = None,
    params_zero: dict = None,
    col_name: str = "transmittance",
    new_col_name: str = "polished",
):
    """
    To automatize cleaning od various data, in this way we do not need to write the same thing over and over:
    - path: path to the file where datas are contained
    - path_zero: path to the file used as reference
    - params_data: if file contains some precaution to be opened
    - params_zero: if file contains some precaution to be opened
    - col_name: name of the column to be refined
    - new_col_name: name of the new column produced
    """
    # Operation to get some names
    base_name = Path(path_data).name.split(".")[0]
    dir_name = Path(path_data).parent
    # Return address
    return_dir = dir_name.joinpath("./ELAB")
    print(return_dir)
    # Change to operating dir, in this way I can use only relative paths
    # os.chdir(dir_name)
    # Check if return adress exists, if not create it
    Path.mkdir(return_dir, parents=True, exist_ok=True)

    # read
    data = pd.read_table(path_data, **params_data)
    zero = pd.read_table(path_zero, **params_zero)
    # make operation
    data[new_col_name] = data[col_name] / zero[col_name]
    data["lambda"] *= 1e-9
    # data["trasm_error"] = data["polished"] * 0.6 / 100 
    data["trasm_error"] = 0.3 * np.sqrt(2) / 100 # Errore dello spettrofotometro: 0.3 per misura, poi sommato
                                                    #in quadratura con se stesso perchè presente sempre anche errore del
                                                    #fondo dell'aria
    # return new data
    data.to_csv(return_dir.as_posix() + "/" + base_name + ".csv", index=False)

#questo prende in input la cartella e usa clean() per 
#pulire tutti i dati presenti nella cartella di input
def clean_dir(
    path: str | Path,
    path_zero: str | Path,
    params_data: dict,
    params_zero: dict,
    col_name: str = "transmittance",
    new_col_name: str = "polished",
):
    """
    To automatize cleaning od various data, in this way we do not need to write the same thing over and over:
    - path: path where datas are contained
    - path_zero: path to the file used as reference
    - params_data: if file contains some precaution to be opened
    - params_zero: if file contains some precaution to be opened
    - col_name: name of the column to be refined
    - new_col_name: name of the new column produced
    """
    for obj in Path(path).iterdir():
        print(obj)
        if obj.is_dir() is False and obj.stem != ".DS_Store":
            clean(obj, path_zero, params_data, params_zero, col_name, new_col_name)


