from pathlib import Path
import torch

class Config:
    # Set default path based on the package structure
    PROJECT_ROOT = Path(__file__).parent
    
    @classmethod
    def get_project_root_dir(cls) -> Path:
        return cls.PROJECT_ROOT