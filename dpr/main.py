import pytorch_lightning as pl
from dpr.model import DensePassageRetrieval
from dataset import PolEvalQADataModule

if __name__ == "__main__":
    model = DensePassageRetrieval()
    PolEvalQA: pl.LightningDataModule = PolEvalQADataModule
    trainer = pl.Trainer()
    trainer.fit(model, PolEvalQA)
