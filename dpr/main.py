import pytorch_lightning as pl
from dpr.model import DensePassageRetrieval
from dataset import QADataModule



def run():
    args = {
        "learning_rate": 1e-5,
        "max_epochs": 2,
        "top_k": 6

    }
    model = DensePassageRetrieval(**args)
    dm: pl.LightningDataModule = QADataModule()
    trainer = pl.Trainer(fast_dev_run=True, max_epochs=2)
    trainer.fit(model, dm)


if __name__ == "__main__":
    run()
