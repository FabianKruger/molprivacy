from moreno_encoders.datamodules.seq2seq import Seq2seqDataModule
from moreno_encoders.models.transformer_vec import SeqToSeqTransformerVector
from moreno_encoders.models.transformer_mat import SeqToSeqTransformerMatrix
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from moreno.config import Config
import sys


class ModelFactory:

    def __init__(self):
        data_dir = Config.get_package_data_dir()
        self.model_dir = data_dir / "encoder_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train_model(self, model_type: str):

        if (self.model_dir / model_type).exists():
            print(
                f"Model already exists. If you want to re-train it, please run `moreno_encoder delete-model {model_type}` first."
            )
            sys.exit()

        datamodule = Seq2seqDataModule()
        datamodule.prepare_data()
        datamodule.setup(stage="fit")
        char2idx = datamodule.char2idx

        assert char2idx is not None
        if model_type == "transformer_vector":
            model = SeqToSeqTransformerVector(char2idx)
        elif model_type == "transformer_matrix":
            model = SeqToSeqTransformerMatrix(char2idx)
        else:
            raise NotImplementedError(f"Model type {model_type} is not supported.")
        logger = CSVLogger(self.model_dir, name=(model_type + "_training"))

        callbacks = []
        callbacks.append(
            ModelCheckpoint(
                dirpath=self.model_dir,
                monitor="average_validation_loss_per_token",
                mode="min",
                filename=model_type,
            )
        )
        callbacks.append(
            EarlyStopping(
                monitor="average_validation_loss_per_token", mode="min", patience=3
            )
        )

        trainer = L.Trainer(callbacks=callbacks, max_epochs=20, logger=logger)
        trainer.fit(model, datamodule)
