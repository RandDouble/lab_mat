"""
Module to be used to clean data in lab,
currentily it cannot be used as standalone application, it must be imported,
It cannot plot any data, if requested it can
maybe in the future, depending on how many application we need to use ti can be expanded
"""

import pandas as pd
# import numpy as np
from pathlib import Path

def clean(path_data : str | Path  , path_zero : str | Path, params_data : dict = None, params_zero : dict = None, col_name : str = "transmittance" , new_col_name : str = "polished" ):
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
    base_name  = Path.basename(path_data).split('.')[0]
    dir_name = Path.dirname(path_data)
    # Return address
    return_dir = Path.relpath("./ELAB")
    # Change to operating dir, in this way I can use only relative paths    
    Path.chdir(dir_name)
    # Check if return adress exists, if not create it
    if return_dir.is_dir()==False:
        Path.mkdir(return_dir)
    # read
    data = pd.read_table(Path.basename(path_data), **params_data)
    zero = pd.read_table(Path.basename(path_zero), **params_zero)
    # make operation
    data[new_col_name] = data[col_name] / zero[col_name]
    # return new data
    data.to_csv(return_dir + base_name +".csv" , index=False)


def clean_dir(path: str | Path, path_zero: str | Path, params_data : dict, params_zero : dict, col_name : str = "transmittance" , new_col_name : str = "polished"):
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
        clean(obj, path_zero, params_data, params_zero, col_name, new_col_name)



# if __name__=="__main__":
#     n = int(input('How many file do you want to clean?'))
    
#     if n > 1:
#         path_data = input("path to data")
#         path_zero = input('Path to file to use as zero')
#         names = input("list of names to give to columns")




