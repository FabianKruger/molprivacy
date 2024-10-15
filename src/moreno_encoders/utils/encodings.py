from typing import List, Dict
import torch
from moreno.config import Config
from moreno_encoders.models.transformer_vec import SeqToSeqTransformerVector
from moreno_encoders.models.transformer_mat import SeqToSeqTransformerMatrix

def generate_vector_encodings(all_smiles: List[str]) -> torch.Tensor:
    data_dir = Config.get_package_data_dir()
    path = (data_dir / "encoder_models" / "transformer_vector.ckpt")
    if not path.exists():
        raise FileNotFoundError("No transformer_vector model found. Run `python -m moreno_encoders install-model transformer_vector` first.")
    model = SeqToSeqTransformerVector.load_from_checkpoint(path)
    model.eval()
    encodings = []
    for smiles in all_smiles:
        encoding = model.get_encoding_from_sequence(smiles)
        encodings.append(encoding.squeeze())
    return torch.stack(encodings).to('cpu')

def generate_matrix_encodings(all_smiles: List[str]) -> List[torch.Tensor]:
    data_dir = Config.get_package_data_dir()
    path = (data_dir / "encoder_models" / "transformer_matrix.ckpt")
    if not path.exists():
        raise FileNotFoundError("No transformer_matrix model found. Run `python -m moreno_encoders install-model transformer_matrix` first.")
    model = SeqToSeqTransformerMatrix.load_from_checkpoint(path)
    model.eval()
    encodings = []
    for smiles in all_smiles:
        encoding = model.get_encoding_from_sequence(smiles)
        encodings.append(encoding.squeeze().to('cpu'))
    return encodings