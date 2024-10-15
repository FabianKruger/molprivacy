from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from typing import Tuple, Dict, List, Optional
import lightning as L
from rdkit import Chem
from rdkit.Chem import SaltRemover
from tqdm import tqdm
import csv
from moreno.config import Config
from pathlib import Path
# from tdc.generation import MolGen

class Seq2seqDataModule(L.LightningDataModule):

    def __init__(self, batch_size = 64, prefetch_factor = 8, num_workers = 8) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers

        self.data_dir = Config.get_package_data_dir()
        self.train_path: Path = self.data_dir / "seq2seq_train.csv"
        self.val_path: Path = self.data_dir / "seq2seq_validation.csv"
        self.char2idx: Optional[Dict[str, int]] = None
        self.idx2char: Optional[Dict[int, str]] = None
        self.train_dataset: Optional[Seq2SeqDataset] = None
        self.val_dataset: Optional[Seq2SeqDataset] = None


    def prepare_data(self) -> None:
        if not (self.train_path.exists() and self.val_path.exists()):
            raise ValueError("Couldnt find training data for seq2seq transformer model. This will be a pain to fix, better raise a GitHub issue.")
            # 
            # data = self.download_data()
            # data = self.clean_data(data)
            # data = data.sample(frac=1, random_state=42).reset_index(drop=True)
            # # split into train and validation data and temporarily save to disc to allow row-wise adding of non-canonical smiles to avoid OOM errors
            # train_data = data.iloc[:int(len(data)*0.9)]
            # validation_data = data.iloc[int(len(data)*0.9):]
            # temp_train_path = self.data_dir / "temp1.csv"
            # temp_val_path = self.data_dir / "temp2.csv"
            # train_data.to_csv(temp_train_path, index=False)
            # validation_data.to_csv(temp_val_path, index=False)
            # self.create_augmentation_file(temp_train_path, self.train_path)
            # self.create_augmentation_file(temp_val_path, self.val_path)

            # # Delete the temporary files after they have been used
            # try:
            #     temp_train_path.unlink()
            #     temp_val_path.unlink()
            # except FileNotFoundError as e:
            #     print(f"Error: {e}")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            assert self.train_path is not None
            assert self.val_path is not None
            self.char2idx, self.idx2char = self.build_vocab_dictionaries(self.train_path, self.val_path)
            self.train_dataset = Seq2SeqDataset(self.train_path, self.char2idx)
            self.val_dataset = Seq2SeqDataset(self.val_path, self.char2idx)
        else:
            raise NotImplementedError
    
    def train_dataloader(self):
        assert self.train_dataset is not None
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=seq2seq_collate_fn, prefetch_factor=self.prefetch_factor, num_workers=self.num_workers)
        return dataloader
    
    def val_dataloader(self):
        assert self.val_dataset is not None
        dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=seq2seq_collate_fn, prefetch_factor=self.prefetch_factor, num_workers=self.num_workers)
        return dataloader  

    # def download_data(self) -> pd.DataFrame:
    #     data = MolGen(name = 'ChEMBL_V29', path=str(self.data_dir))
    #     data = pd.DataFrame(data.get_data())
    #     return data
    
    # def clean_data(self, data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    #     # clean data
    #     data.dropna(inplace=True)
    #     indices_to_drop = []

    #     for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    #         smiles = row['smiles']
    #         try:
    #             mol = Chem.MolFromSmiles(smiles)
    #             if mol is None:
    #                 print(f"Failed to parse SMILES: {smiles}")
    #                 indices_to_drop.append(index)
    #                 continue

    #             Chem.SanitizeMol(mol)
    #             mol = SaltRemover.SaltRemover().StripMol(mol, dontRemoveEverything=True)
    #             if len(Chem.GetMolFrags(mol)) > 1:
    #                 if verbose:
    #                     print(
    #                         f"Multiple fragments for {smiles}. Dropped since label assignment is not clear."
    #                     )
    #                 indices_to_drop.append(index)
    #                 continue
    #             # make canonical smiles
    #             can_smiles = Chem.MolToSmiles(mol, canonical=True)
    #             assert isinstance(
    #                 can_smiles, str
    #             ), f"Canonical SMILES is not a string:{smiles} {can_smiles}"
    #             assert (
    #                 can_smiles != ""
    #             ), f"Canonical SMILES is empty for smiles: {smiles} {can_smiles}"
    #             data.at[index, "smiles"] = can_smiles
    #         except Exception as e:
    #             print(f"Failed to clean molecule: {smiles}. It will be dropped. Error: {str(e)}")
    #             indices_to_drop.append(index)

    #     # Drop problematic rows
    #     print(f"Dropping {len(indices_to_drop)} rows.")
    #     data.drop(indices_to_drop, inplace=True)

    #     # Sort the DataFrame, find dublicates, and filter them out (keeping the first occurence)
    #     data = data.sort_values(['smiles'])
    #     duplicates = data.duplicated(subset=['smiles'], keep='first')
    #     data = data[~duplicates]
    #     return data

    # def create_augmentation_file(self, raw_data_path: Path, result_path: Path) -> None:
    #     # Open the input file in read mode and the output file in write mode
    #     with open(raw_data_path, 'r') as input_file, open(result_path, 'w', newline='') as output_file:
    #         # Create a csv reader and writer
    #         reader = csv.reader(input_file)
    #         writer = csv.writer(output_file)
            
    #         # Create a progress bar with tqdm
    #         rows = list(reader)
    #         pbar = tqdm(rows, total=len(rows))
            
    #         # flag for first row
    #         is_first_row = True
            
    #         # Iterate over each row in the input file
    #         for row in pbar:
                
    #             # Treat first row differently, because it contains the column names
    #             if is_first_row:
    #                 writer.writerow(['non_canonical_smiles', 'canonical_smiles'])
    #                 is_first_row = False
    #                 continue
                
    #             # Get the string from the row
    #             original_smiles = row[0]

    #             # Transform the string canonical_smiles_generation
    #             canonical_smiles = self.canonical_smiles_generation(original_smiles)

    #             max_length = 200
    #             if len(canonical_smiles)<=max_length:
    #                 # Augment the string
    #                 non_canonical_smiles_list = self.non_canonical_smiles_generation(smiles=original_smiles, amount=10, add_canoncial=True)

    #                 # Write each augmented string to the output file on its own row, paired with transformed string
    #                 for non_canonical_smiles in non_canonical_smiles_list:
    #                     if len(non_canonical_smiles)<=max_length:
    #                         writer.writerow([non_canonical_smiles, canonical_smiles])
    #         pbar.close()

    # def canonical_smiles_generation(self, smiles: str) -> str:
    #     """Generate canonical SMILES from SMILES"""
        
    #     # raising errors for faulty input
    #     if not isinstance(smiles, str):
    #         raise TypeError("smiles must be a string")
    #     molecule = Chem.MolFromSmiles(smiles)
    #     if molecule is None:
    #         raise ValueError(f"Could not convert smiles to molecule. Make sure {smiles} is a valid SMILES string")
        
    #     # generate canonical smiles
    #     canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
        
    #     # return canonical smiles
    #     return canonical_smiles
    
    
    # def non_canonical_smiles_generation(self, smiles: str, amount: int = 10, add_canoncial: bool = True) -> List[str]:
    #     """Generate list of amount of random noncanonical SMILES from canonical SMILES.
    #     If add_canoncial is True, the canonical SMILES will be added as last element of the list."""
        
    #     # raising errors for faulty input
    #     if not isinstance(smiles, str):
    #         raise TypeError("smiles must be a string")
    #     molecule = Chem.MolFromSmiles(smiles)
    #     if molecule is None:
    #         raise ValueError(f"Could not convert smiles to molecule. Make sure {smiles} is a valid SMILES string")
        
    #     # generate noncanonical smiles
    #     non_canonical_smiles = Chem.MolToRandomSmilesVect(molecule, amount)  

    #     # give warning if we could not generate the amount of unique noncanonical smiles
    #     if len(non_canonical_smiles) < amount:
    #         print(f"Could not generate the amount of unique noncanonical smiles. Found only {len(non_canonical_smiles)} unique noncanonical smiles for {smiles}")        
        
    #     # add canonical smiles as last element if desired
    #     if add_canoncial:
    #         canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
    #         non_canonical_smiles.append(canonical_smiles)
        
    #     # return noncanonical smiles list
    #     return non_canonical_smiles


    def build_vocab_dictionaries(self, *csv_file_paths: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Build a vocabulary for character-level sequence-to-sequence model from multiple csv files (e.g., train, val, test).
        Csv files should have two columns, one for input sequences and one for target sequences.
        
        Parameters:
        *csv_files: str: Names of csv files to build the vocabulary from.
        
        Returns:
        char2idx: Dict[str, int]: A dictionary mapping each unique character to a unique integer.
        idx2char: Dict[int, str]: A dictionary mapping each unique integer back to its corresponding character.
        """
        # Set for unique characters
        unique_chars = set()
        
        # For each csv file
        for csv_file in csv_file_paths:
            print(str(csv_file))
            data = pd.read_csv(csv_file)    
            # Extract unique characters from the input and target sequences and add them to the set
            file_chars = set(''.join(data.iloc[:, 0]) + ''.join(data.iloc[:, 1]))
            unique_chars.update(file_chars)
        
        # add a start of sequence token, end of sequence token, padding token, and unknown token
        PAD = 0
        SOS = 1
        EOS = 2
        UNK = 3
        # Map each unique character to a unique integer
        char2idx = {char: idx + 4 for idx, char in enumerate(unique_chars)}
        char2idx['<PAD>'] = PAD
        char2idx['<SOS>'] = SOS
        char2idx['<EOS>'] = EOS
        char2idx['<UNK>'] = UNK
        # Map each unique character to a unique integer
        # Create a reverse mapping from integers to characters
        idx2char = {idx: char for char, idx in char2idx.items()}
        
        return char2idx, idx2char   



class Seq2SeqDataset(Dataset):
    """
    Dataset for seq2seq models.
    Works for csv files with two columns, one for input sequences and one for target sequences.
    """
    
    def __init__(self, csv_file_path: Path, character2index: Dict[str, int]) -> None:
        self.data = pd.read_csv(csv_file_path)
        self.char2idx = character2index
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_sequence = self.data.iloc[idx, 0]
        target_sequence = self.data.iloc[idx, 1]
        assert isinstance(input_sequence, str)
        assert isinstance(target_sequence, str)
        # Convert characters to integers using the char2idx mapping, add end and beginning of sequence tokens and make it a tensor
        input_sequence = torch.tensor([self.char2idx['<SOS>']]+[self.char2idx.get(char, self.char2idx['<UNK>']) for char in input_sequence]+[self.char2idx['<EOS>']], dtype=torch.long)
        # append one PAD token to the target which will be removed later when right shifting the target sequence
        target_sequence = torch.tensor([self.char2idx['<SOS>']]+[self.char2idx.get(char, self.char2idx['<UNK>']) for char in target_sequence]+[self.char2idx['<EOS>']] + [self.char2idx['<PAD>']], dtype=torch.long)       
        return input_sequence, target_sequence
    

def seq2seq_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for the dataloader.
    Adds padding to make all input sequences in a batch the same length.
    Adds padding to make all target sequences in a batch the same length.
    Returns input batch and target batch as tensors.
    Returned tensors are in the shape batch_size x max_sequence_length.
    """
    # extract input and target sequences from the batch
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    
    # pad input sequence with pytorch pad_sequence function
    data = pad_sequence(data, batch_first=True, padding_value=0)
    
    # pad target sequence with pytorch pad_sequence function
    target = pad_sequence(target, batch_first=True, padding_value=0)
    return data, target