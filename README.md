# Training data privacy assessment for molecular property prediction

Python package to assess how much information somebody can deduct from a neural network trained on some (confidential) training data for molecular property prediction.

## Getting Started

These instructions will help you install the python package and conduct a privacy assessments on data for molecular property prediction.

### Prerequisites

You need to have an environment with python version 3.12. This can be created for example using conda.

```
$ conda create -n privacy_env python=3.12 
$ conda activate privacy_env 
```

### Installation

Package can be installed from PyPI with pip

```
$ pip install molprivacy
```

## Usage

The package can be run via the command line or by importing the privacy_test function into your script.

### Command line interface
You can run the privacy test directly from the command line using the `privacytest` command after installing the package.

```
$ privacytest --representation REPRESENTATION --result_folder RESULT_FOLDER [options]
```

#### Required Arguments
`--representation`: Specifies the molecular representation to use. Choices are:
- `ECFP4`
- `ECFP6`
- `MACCS`
- `graph`
- `rdkit`
- `transformer_vector`
- `transformer_matrix`

`--result_folder`: Path to the folder where the results will be stored.

#### Optional Arguments
`--dataset`: Specifies the dataset to use. Choices are:
- `ames` (default)
- `herg`
- `del`
- `bbb`
- `file` (use your own dataset; requires `--dataset_path`)

`--dataset_path`: Path(s) to your custom dataset file(s). Required if `--dataset file` is selected. The dataset must have a 'smiles' column and a binary 'label' column.

`--split`: Split ratios for training, validation, and testing datasets. Provide three float values that sum to 1.0. Testing dataset will be used for privacy assessment.
Default: `0.45 0.1 0.45`

`--hyperparameter_optimization_time`: Time in seconds allocated for hyperparameter optimization during model training.
Default: `600`

`--attack_data_fraction`: Fraction of data to use for the Reverse Model Inversion Attack (RMIA). Reduce this value if the RMIA attack runs out of memory.
Default: `1.0`

#### Examples
Run with default dataset and parameters:
```
$ privacytest --representation ECFP4 --result_folder "home/results"
```
Use a custom dataset:
```
$ privacytest --representation MACCS --result_folder "home/results" --dataset file --dataset_path "home/data/my_dataset.csv"
```

#### Help
For a full list of available options and detailed descriptions, run:
```
$ privacytest --help
```

### Import into a script
In addition to running the privacy test from the command line, you can directly use the `privacy_test` function in your Python scripts. This allows for seamless integration into your workflows and the ability to to customize parameters programmatically.

#### Import the function
```
from privacytest.__main__ import privacy_test
```
#### Example usage
```
from privacytest.__main__ import privacy_test
# Import your custom representation function (needs to have a smiles string as input and a tuple of the encoding vector and the vectors dimension as an output)
from mycode import my_custom_representation_function

# Define privacy test parameters
representation = 'custom'
result_folder = 'home/results'
dataset = 'ames'
split = [0.45, 0.1, 0.45]
hyperparameter_optimization_time = 600
attack_data_fraction = 1.0
custom_representation_function = my_custom_representation_function

# Run the privacy test with custom representation
privacy_test(
    representation=representation,
    result_folder=result_folder,
    dataset=dataset,
    split=split,
    hyperparameter_optimization_time=hyperparameter_optimization_time,
    attack_data_fraction=attack_data_fraction,
    custom_representation_function=custom_representation_function
)
```

## Privacy Test Output

When you run the privacy test, the results are saved in the `result_folder` you specified. The folder structure contains the following files and directories:

#### `privacy/`
- **results/**: This folder contains the outputs related to privacy performance for both the LiRA and RMIA privacy attacks.
  - **lira/**:
    - **privacy_performance_overview.txt**: A text file summarizing the results of the LiRA privacy attack.
    - **privacy_performance.pdf**: ROC curve for the LiRA attack (positives are training data samples).
    - **ROC.csv**: CSV file containing data of ROC curve.
  
  - **rmia/**:
    - **privacy_performance_overview.txt**: A text file summarizing the results of the RMIA privacy attack.
    - **privacy_performance.pdf**: ROC curve for the RMIA attack.
    - **ROC.csv**: CSV file containing data of ROC curve.
  
  - **true_positives_at_FPR0/**:
    - **lira.csv**: CSV file that contains all the chemical structures that LiRA could identify at a False Positive Rate of 0.
    - **rmia.csv**: CSV file that contains all the chemical structures that RMIA could identify at a False Positive Rate of 0.

#### `model/`
- **model_performance.pdf**: ROC curve of the model performance in the binary classification task.
- **model.ckpt**: The checkpoint file for the final model that was trained with the optimized hyperparameters.
- **optimization.db**: Database containing the optimization history during hyperparameter tuning.
- **optimized_hyperparameters.yaml**: A YAML file that contains the optimized hyperparameters after the search process.


#### `data_dir/`
- **train.csv**, **validation.csv**, **test.csv**: These files contain the datasets used for training, validation, and testing. Will vary when re-running the package since the split is random.

#### `model_config.yaml`
- A configuration file that summarizes all model related configurations.

#### `privacy_config.yaml`
- A configuration file that summarizes all membership inference attack related configurations.


## Citation

This repository is part of the paper "Publishing neural networks in drug discovery compromises training data privacy". \\
Pre-print: TODO \\
Bibtex: TODO



