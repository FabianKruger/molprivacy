from pathlib import Path
import torch

class Config:
    # Set default path based on the package structure
    PROJECT_ROOT = Path(__file__).parent
    PACKAGE_DATA_DIR = PROJECT_ROOT / "data"
    DATA_DIR = PROJECT_ROOT / "data"
    DIR = Path.cwd()
    # Set default pos_weights for binary classification (to counteract imbalanced datasets)
    POS_WEIGHTS = torch.tensor([1])

    @classmethod
    def set_data_dir(cls, new_path: Path | str):
        cls.DATA_DIR = Path(new_path)

    @classmethod
    def get_data_dir(cls) -> Path:
        return cls.DATA_DIR
    
    @classmethod
    def set_directory(cls, new_path: Path | str):
        cls.DIR = Path(new_path)

    @classmethod
    def get_directory(cls) -> Path:
        return cls.DIR

    @classmethod
    def set_pos_weights(cls, new_weights: torch.Tensor):
        cls.POS_WEIGHTS = new_weights

    @classmethod
    def get_pos_weights(cls) -> torch.Tensor:
        return cls.POS_WEIGHTS
    
    @classmethod
    def get_package_data_dir(cls) -> Path:
        return cls.PACKAGE_DATA_DIR
    
    @classmethod
    def get_project_root_dir(cls) -> Path:
        return cls.PROJECT_ROOT