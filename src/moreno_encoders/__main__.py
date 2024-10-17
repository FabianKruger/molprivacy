import argparse
import shutil
from moreno_encoders.factories.model import ModelFactory
from typing import Optional
from moreno.config import Config
import os


def install_model(model_type: str):
    factory = ModelFactory()
    factory.train_model(model_type)
    print(f"Model {model_type} installed successfully")


def delete_models(model_type: Optional[str] = None):
    # Delete all models. Ask for confirmation first. If model_type is specified, delete the specified model.
    data_dir = Config.get_data_dir()
    model_dir = data_dir / "encoder_models"

    if model_type:
        model_path = model_dir / (model_type + ".ckpt")
        log_path = model_dir / (model_type + "_training")
        if model_path.exists():
            confirmation = input(
                f"Are you sure you want to delete the model {model_type}? This action cannot be undone. (yes/no): "
            )
            if confirmation.lower() == "yes":
                os.remove(model_path)
                if log_path.exists():
                    shutil.rmtree(log_path)
                print(f"Model {model_type} and its training logs deleted successfully")
            else:
                print("Deletion cancelled")
        else:
            print(f"Model {model_type} does not exist")
    else:
        confirmation = input(
            "Are you sure you want to delete all models? This action cannot be undone. (yes/no): "
        )
        if confirmation.lower() == "yes":
            shutil.rmtree(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)  # Recreate the model directory
            print("All models and their training logs deleted successfully")
        else:
            print("Deletion cancelled")


def main():
    parser = argparse.ArgumentParser(description="MoReNO encoder CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Define the install-model command
    install_parser = subparsers.add_parser("install-model", help="Install the model")
    install_parser.add_argument(
        "model_type",
        choices=["transformer_vector", "transformer_matrix"],
        help="Type of the model to install",
    )
    install_parser.set_defaults(func=install_model)

    # Define the delete-models command
    delete_parser = subparsers.add_parser("delete-models", help="Delete models")
    delete_parser.add_argument(
        "model_type",
        nargs="?",
        choices=["transformer_vector", "transformer_matrix"],
        help="Type of the model to delete. If not specified, all models will be deleted",
    )
    delete_parser.set_defaults(func=delete_models)

    args = parser.parse_args()

    if args.command:
        if args.command == "install-model":
            args.func(args.model_type)
        elif args.command == "delete-models":
            args.func(args.model_type)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
