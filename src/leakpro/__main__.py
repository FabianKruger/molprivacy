"""Main script to run LEAKPRO on a target model."""

import logging
import random
import time
from pathlib import Path

import numpy as np
import yaml
from torch import manual_seed
from typing import Literal


from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.reporting.utils import prepare_priavcy_risk_report
from leakpro.user_inputs.moreno_input_handler import MorenoInputHandler
from leakpro.config import Config

def setup_log(name: str, log_dir: Path, save_file: bool=True) -> logging.Logger:
    """Generate the logger for the current run.

    Args:
    ----
        name (str): Logging file name.
        save_file (bool): Flag about whether to save to file.

    Returns:
    -------
        logging.Logger: Logger object for the current run.

    """
    my_logger = logging.getLogger(name)
    my_logger.setLevel(logging.INFO)
    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

    # Console handler for output to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    my_logger.addHandler(console_handler)

    if save_file:
        # Ensure the log directory exists
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Set the log file path
        filename = log_dir / f"log_{name}.log"
        log_handler = logging.FileHandler(filename, mode="w")
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(log_format)
        my_logger.addHandler(log_handler)


    return my_logger


def main(
    representation: Literal["ECFP4", "ECFP6", "MACCS", "graph", "rdkit", "transformer_vector", "transformer_matrix"],
    result_folder: str | Path,
    attack_data_fraction: float = 1.0
    ) -> None:


    root = Config.get_project_root_dir()
    args = root / "config" / "moreno_audit.yaml"
    
    with open(args, "rb") as f:
        configs = yaml.safe_load(f)

    result_folder = Path(result_folder)

    configs["moreno"]["representation"] = representation
    configs["moreno"]["hyperparameters_path"] = result_folder / "model" / "training" / "version_0" / "hparams.yaml"
    configs["moreno"]["model_path"] = result_folder / "model" / "model.ckpt"
    configs["data"]["train_data_path"] = result_folder / "data_dir" / "train.csv"
    configs["data"]["test_data_path"] = result_folder / "data_dir" / "test.csv"
    configs["shadow_model"]["storage_path"] = result_folder / "privacy" / "shadow_models"
    configs["audit"]["report_log"] = result_folder / "privacy" / "results"
    configs["audit"]["config_log"] = result_folder / "privacy" / "results" / "configs"
    # define these for later train data extraction TODO: seems unnecessary, fix it
    configs["audit"]["attack_list"]["rmia"]["train_data_path"] = result_folder / "data_dir" / "train.csv"
    configs["audit"]["attack_list"]["rmia"]["representation"] = representation
    configs["audit"]["attack_list"]["rmia"]["report_log"] = result_folder / "privacy" / "results"
    configs["audit"]["attack_list"]["lira"]["train_data_path"] = result_folder / "data_dir" / "train.csv"
    configs["audit"]["attack_list"]["lira"]["representation"] = representation
    configs["audit"]["attack_list"]["lira"]["report_log"] = result_folder / "privacy" / "results"

    # Setup logger
    logger = setup_log("LeakPro", log_dir= result_folder / "privacy", save_file=True)

    # define attack data fraction 
    configs["audit"]["attack_list"]["rmia"]["attack_data_fraction"] = attack_data_fraction
    # Set the random seed, log_dir and inference_game
    manual_seed(configs["audit"]["random_seed"])
    np.random.seed(configs["audit"]["random_seed"])
    random.seed(configs["audit"]["random_seed"])

    # Create directory to store results
    report_dir = configs['audit']['report_log']
    report_dir.mkdir(parents=True, exist_ok=True)

    # Create user input handler
    handler = MorenoInputHandler(configs=configs, logger=logger)

    attack_scheduler = AttackScheduler(handler)
    audit_results = attack_scheduler.run_attacks()

    for attack_name in audit_results:
        logger.info(f"Preparing results for attack: {attack_name}")

        prepare_priavcy_risk_report(
                audit_results[attack_name]["result_object"],
                configs["audit"],
                save_path=f"{report_dir}/{attack_name}",
            )
    # ------------------------------------------------
    # Save the configs and user_configs
    config_log_path = result_folder / "privacy_config.yaml"
    with open(config_log_path, 'w') as file:
        yaml.dump(configs, file, default_flow_style=False)