from typing import List, Optional, Literal, Callable
from pathlib import Path
import yaml
import copy
from moreno.factories.datasets import DataFactory
from moreno.factories.models import ModelFactory
from moreno.config import Config
import numpy as np
import moreno.custom_representation

# Updated main function with Literal for specific choices
def main(
    task: Literal["train", "download"],
    dataset: Literal["ames", "herg", "del", "bbb", "file"] = "ames",
    split: Optional[List[float]] = None,
    representations: Optional[List[Literal["custom", "ECFP4", "ECFP6", "MACCS", "graph", "rdkit", "transformer_vector", "transformer_matrix"]]] = None,
    hyperparameter_optimization_time: int = 600,
    result_folder: Optional[str] = None,
    save_csv_in_results: bool = True,
    dataset_path: Optional[List[str]] = None,
    custom_representation_function: Optional[Callable[[np.ndarray, int], None]] = None, 
) -> None:
    """
    Main function to handle training and dataset preparation with specific options.

    :param task: The task to execute, either 'train' or 'download'
    :param dataset: The dataset to use for training. Choices: 'ames', 'herg', 'bbb', 'file'
    :param split: List of three floats representing the dataset split proportions for train/validation/test
    :param representations: List of representations to test. Choices: 'ECFP4', 'ECFP6', 'MACCS', 'graph', 'rdkit', 'transformer_vector', 'transformer_matrix'
    :param hyperparameter_optimization_time: Time in seconds to search for hyperparameters for each representation.
    :param result_folder: Folder path containing the results (trained models, data, etc.)
    :param save_csv_in_results: Whether to save the CSV files in the results folder
    :param dataset_path: List of file paths to the dataset, required if dataset is 'file'
    :param custom_representation_function: Custom representation function for custom representations. Takes list of smiles as input and ouputs a Tuple of 2 elements. First element is an array of arrays(vectors) and second element is the input dimension (Vector size) for the MLP.
    """

    # If no split provided, set default
    if split is None:
        split = [0.7, 0.1, 0.2]

    if "custom" in representations:
        if custom_representation_function is None:
            raise ValueError("Custom representation function was not provided but 'custom' representation was chosen.")
        moreno.custom_representation.convert_vector = custom_representation_function
        
    # Create an args-like dictionary to hold all the provided parameters
    args = {
        'command': task,
        'dataset': dataset,
        'split': split,
        'representations': representations,
        'hyperparameter_optimization_time': hyperparameter_optimization_time,
        'result_folder': result_folder,
        'save_csv_in_results': save_csv_in_results,
        'dataset_path': dataset_path
    }

    # Dispatch based on the task
    if task == "train":
        train_models(args)
    else:
        raise ValueError("Invalid task. Use 'train'.")

def train_models(args: dict) -> None:
    """
    Train models based on the provided arguments.
    
    :param args: Dictionary of parsed arguments
    """
    # Save args to dict for saving as yaml file later
    args_dict = copy.deepcopy(args)

    # Checking for dataset_path
    if args['dataset'] == "file":
        if args['dataset_path'] is None:
            raise ValueError("--dataset_path is required when --dataset is 'file'")
        if isinstance(args['dataset_path'], str):
            args['dataset_path'] = [args['dataset_path']]
        if len(args['dataset_path']) not in [1, 2, 3]:
            raise ValueError("--dataset_path must contain 1, 2, or 3 arguments")

    # Set up directories
    setup_directories(args)

    # Save execution config
    save_execution_config(args_dict)

    # Train models
    for representation in args['representations']:
        data_factory = DataFactory(
            dataset=args['dataset'], split=args['split'], representation=representation, paths=args['dataset_path']
        )
        data_module = data_factory.create_datamodule()
        model_factory = ModelFactory(
            datamodule=data_module,
            hyperparameter_optimization_time=args['hyperparameter_optimization_time'],
        )
        model_factory.create_optimizer()
        model_factory.create_model()
        model_factory.test_model()

def setup_directories(args: dict) -> None:
    """
    Set up directories for storing results and data.
    
    :param args: Dictionary of parsed arguments
    """
    if args['result_folder'] is not None:
        result_dir = Path(args['result_folder'])
        if not (result_dir.exists() and result_dir.is_dir()):
            raise FileNotFoundError(f"The result folder {result_dir} does not exist or is not a directory.")
        Config.set_directory(result_dir)
    else:
        args['result_folder'] = Config.get_directory()  # default is current working directory
    if args['save_csv_in_results']:
        data_dir = Config.get_directory() / "data_dir"
        data_dir.mkdir(exist_ok=True)
        Config.set_data_dir(data_dir)

def save_execution_config(args_dict: dict) -> None:
    """
    Save the execution configuration to a YAML file.
    
    :param args_dict: Dictionary of parsed arguments to be saved
    """
    base_yaml_file_path = Config.get_directory() / "model_config"
    yaml_file_path = base_yaml_file_path.with_suffix(".yaml")
    counter = 1
    while yaml_file_path.exists():
        yaml_file_path = base_yaml_file_path.with_name(f"{base_yaml_file_path.stem}({counter})").with_suffix(".yaml")
        counter += 1
    with open(yaml_file_path, 'w') as file:
        yaml.dump(args_dict, file, default_flow_style=False)

if __name__ == "__main__":
    main()