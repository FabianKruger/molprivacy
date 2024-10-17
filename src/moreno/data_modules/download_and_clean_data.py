# # datasets should be included in the Github repository. This is just in case someone wants to check how the data files were created.
# from tdc.single_pred import ADME, Tox
# from moreno.config import Config
# from moreno.utils.clean_data import clean_smiles_df
# from moreno.utils.csv_file_operations import save_csv
# from typing import Literal
# import pandas as pd

# def download_data(dataset_name: Literal["bbb", "ames", "herg"]) -> pd.DataFrame:
#     """Download data from Therapeutics Data Commons. Datasets are under CC BY 4.0 license (last checked 2024-08-27).

#     Args:
#         dataset_name: Dataset to download. Accepts either "bbb" or "ames" or "herg".

#     Raises:
#         ValueError: Dataset name is not implemented.

#     Returns:
#         pd.DataFrame: Pandas dataframe with columns "Drug", which contains smiles strings and "Y", which contains the corresponding binary labels.
#     """
#     data_dir = Config.get_data_dir()
#     if dataset_name == "bbb":
#         data = ADME(name="BBB_Martins", path=str(data_dir))
#     elif dataset_name == "ames":
#         data = Tox(name="AMES", path=str(data_dir))
#     elif dataset_name == "herg":
#         data = Tox(name="herg_central", label_name="hERG_inhib", path=str(data_dir))
#     else:
#         raise ValueError(f"Invalid dataset name: {dataset_name}")

#     return pd.DataFrame(data.get_data())


# def run_data_preparation() -> None:
#     """Downloads and saves the datasets "bbb", "ames", and "herg" from Therapeutics Data Commons if they do not exist in the data directory.
#     """
#     data_dir = Config.get_data_dir()
#     for dataset in ["bbb", "ames", "herg"]:
#         data_path = data_dir / f"{dataset}.csv"
#         if not data_path.exists():
#             data = download_data(dataset)
#             data.rename(columns={"Drug": "smiles", "Y": "label"}, inplace=True)
#             data = clean_smiles_df(data)
#             save_csv(data, f"{dataset}.csv")
