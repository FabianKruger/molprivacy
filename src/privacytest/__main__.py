from moreno.__main__ import main as momain
from leakpro.__main__ import main as lemain
from typing import Optional, List, Literal, Callable
import numpy as np
import argparse


def privacy_test(
    representation: Literal[
        "ECFP4",
        "ECFP6",
        "MACCS",
        "graph",
        "rdkit",
        "transformer_vector",
        "transformer_matrix",
        "custom",
    ],
    result_folder: str,
    dataset: Literal["ames", "herg", "del", "bbb", "file"] = "ames",
    dataset_path: Optional[List[str]] = None,
    split: List[float] = [0.45, 0.1, 0.45],
    hyperparameter_optimization_time: int = 600,
    attack_data_fraction: float = 1.0,
    custom_representation_function: Optional[Callable[[np.ndarray, int], None]] = None,
) -> None:
    # Run moreno
    momain(
        task="train",
        dataset=dataset,
        split=split,
        representations=[representation],
        hyperparameter_optimization_time=hyperparameter_optimization_time,
        result_folder=result_folder,
        save_csv_in_results=True,
        dataset_path=dataset_path,
        custom_representation_function=custom_representation_function,
    )
    # Run LeakPro
    if dataset == "herg":
        attack_data_fraction = 0.2
    lemain(
        representation=representation,
        result_folder=result_folder,
        attack_data_fraction=attack_data_fraction,
    )


def main():
    parser = argparse.ArgumentParser(description="Run Moreno and LeakPro")

    parser.add_argument(
        "--representation",
        type=str,
        choices=[
            "ECFP4",
            "ECFP6",
            "MACCS",
            "graph",
            "rdkit",
            "transformer_vector",
            "transformer_matrix",
            "custom",
        ],
        help="Representation type.",
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        help="Path to the folder where the result will be stored.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ames", "herg", "del", "bbb", "file"],
        default="ames",
        help="Dataset to use. Choices are bbb, ames, del, herg, file (default: ames). If choice is file, --dataset_path needs to be specified.",
    )
    parser.add_argument("--dataset_path", nargs="+", help="Path(s) to dataset file(s)")
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.45, 0.1, 0.45],
        help="Split ratios as three float values (default: 0.45 0.1 0.45)",
    )
    parser.add_argument(
        "--hyperparameter_optimization_time",
        type=int,
        default=600,
        help="Time to spend searching for optimized hyperparameter in seconds (default: 600)",
    )
    parser.add_argument(
        "--attack_data_fraction",
        type=float,
        default=1.0,
        help="Attack data fraction. Reduce if RMIA attack runs out of memory (default: 1.0).",
    )

    args = parser.parse_args()

    privacy_test(
        representation=args.representation,
        result_folder=args.result_folder,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        split=args.split,
        hyperparameter_optimization_time=args.hyperparameter_optimization_time,
        attack_data_fraction=args.attack_data_fraction,
    )


if __name__ == "__main__":
    main()

    # from rdkit import Chem
    # def smiles_to_tensor(smiles_list):
    # # Define the atoms of interest
    #     atom_symbols = ['C', 'O', 'N', 'S', 'P']

    #     # List to store the tensors
    #     array_list = []

    #     for smiles in smiles_list:
    #         # Parse the SMILES string using RDKit
    #         mol = Chem.MolFromSmiles(smiles)

    #         # Create a list to hold the counts of each atom type
    #         atom_counts = [0] * len(atom_symbols)

    #         # Count the occurrences of each atom in the molecule
    #         for atom in mol.GetAtoms():
    #             symbol = atom.GetSymbol()
    #             if symbol in atom_symbols:
    #                 atom_counts[atom_symbols.index(symbol)] += 1

    #         # Convert the atom counts to a tensor
    #         array_list.append(np.array(atom_counts))

    #     # Convert the list of tensors to a NumPy array
    #     return np.array(array_list), 5

    # privacy_test(representation="rdkit", result_folder="/home/khcq385/MOPADD/temp", dataset="del", hyperparameter_optimization_time=120)
