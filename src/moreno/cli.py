import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Moreno: Create models with optimized hyperparameters for specified representations."
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train models with optimized hyperparameters')
    _add_train_arguments(train_parser)

    # Download command
    subparsers.add_parser('download', help='Download and prepare all datasets')

    return parser.parse_args()

def _add_train_arguments(parser):
    parser.add_argument(
        "--dataset", type=str, default="ames", help="The dataset to train the models on. Built in options are: ames, herg, bbb, file"
    )
    parser.add_argument(
        "--split", type=float, nargs=3, default=[0.7, 0.1, 0.2], help="The split of the dataset. First index is the train proportion, second index is the validation proportion, third index is the test proportion"
    )
    parser.add_argument(
        "--representations", nargs="+", help="Representations that should be tested. Allows ECFP4, ECFP6, MACCS, graph, rdkit, transformer_vector, transformer_matrix."
    )
    parser.add_argument(
        "--hyperparameter_optimization_time",
        type=int,
        default=600,
        help="Time in seconds to search for hyperparameters for each representation",
    )
    parser.add_argument(
        "--result_folder", type=str, default=None, help="Folder path containing all the results (trained model, hyperparameter optimization dabatase, final model training data). Defaults to the current working directory."
    )
    parser.add_argument(
        "--save_csv_in_results", type=bool, default=False, help="If true, the csv files containing the cleaned data will be saved in the results folder, otherwise they will be saved in the package folder."
    )
    parser.add_argument(
        "--dataset_path", type=str, nargs='+', help="Path(s) to the dataset file(s). Required if --dataset is 'file'."
    )